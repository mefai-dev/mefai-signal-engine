"""
Mefai Signal Engine - RL Position Sizer

Deep Q-Network (Double DQN) for optimal position sizing.
Uses experience replay, target network soft updates, and
epsilon-greedy exploration.
"""

import logging
import random
from collections import deque
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)

POSITION_ACTIONS = [0.0, 0.25, 0.50, 0.75, 1.0]
STATE_DIM = 7
N_ACTIONS = len(POSITION_ACTIONS)
REPLAY_BUFFER_SIZE = 100_000
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.005
LR = 1e-4
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_STEPS = 10_000


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Experience replay buffer with uniform sampling."""

    def __init__(self, capacity: int = REPLAY_BUFFER_SIZE):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, transition: Transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


class DQNetwork(nn.Module):
    """
    Deep Q-Network for position sizing.

    Input state vector (7 dims):
        [current_position, unrealized_pnl, volatility,
         signal_strength, regime, drawdown, win_rate]

    Output: Q-values for each position size action
        [0%, 25%, 50%, 75%, 100%]
    """

    def __init__(self, state_dim: int = STATE_DIM, n_actions: int = N_ACTIONS):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.network(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values


class RLPositionSizer:
    """
    Double DQN position sizer with experience replay and soft target updates.

    State representation:
        - current_position: current position fraction [0, 1]
        - unrealized_pnl: normalized unrealized profit/loss
        - volatility: current market volatility (normalized)
        - signal_strength: signal confidence [0, 1]
        - regime: market regime encoding [0=bull, 0.33=bear, 0.66=sideways, 1=high_vol]
        - drawdown: current drawdown from peak [0, 1]
        - win_rate: recent win rate [0, 1]

    Actions: position size as fraction of max - [0%, 25%, 50%, 75%, 100%]

    Reward: risk-adjusted return
        reward = pnl / max(volatility, 0.01) - drawdown_penalty
    """

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.policy_net = DQNetwork().to(self.device)
        self.target_net = DQNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.replay_buffer = ReplayBuffer()

        self.epsilon = EPSILON_START
        self.steps = 0
        self.training_losses: List[float] = []

        self.state_mean = np.zeros(STATE_DIM)
        self.state_std = np.ones(STATE_DIM)

    @staticmethod
    def compute_reward(
        pnl: float, volatility: float, drawdown: float,
        drawdown_penalty_weight: float = 2.0
    ) -> float:
        """
        Compute risk-adjusted reward.

        Args:
            pnl: Profit/loss from the trade
            volatility: Current market volatility
            drawdown: Current drawdown fraction [0, 1]
            drawdown_penalty_weight: Weight for drawdown penalty

        Returns:
            Risk-adjusted reward value
        """
        risk_adjusted_return = pnl / max(volatility, 0.01)
        drawdown_penalty = drawdown_penalty_weight * drawdown ** 2
        return risk_adjusted_return - drawdown_penalty

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state vector (7 dims)
            training: Whether to use exploration

        Returns:
            Action index (0-4)
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, N_ACTIONS - 1)

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax(dim=1).item()

    def get_position_size(self, state: np.ndarray) -> Dict:
        """
        Get optimal position size for given state.

        Args:
            state: State vector [position, pnl, vol, signal, regime, dd, wr]

        Returns:
            Dict with position_pct, action_index, q_values, confidence
        """
        state_norm = (state - self.state_mean) / (self.state_std + 1e-8)
        state_tensor = (
            torch.tensor(state_norm, dtype=torch.float32).unsqueeze(0).to(self.device)
        )

        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]

        action = int(np.argmax(q_values))
        position_pct = POSITION_ACTIONS[action]

        q_range = q_values.max() - q_values.min()
        if q_range > 0:
            confidence = float((q_values[action] - q_values.min()) / q_range)
        else:
            confidence = 0.5

        return {
            "position_pct": position_pct,
            "action_index": action,
            "q_values": q_values.tolist(),
            "confidence": confidence,
        }

    def store_transition(
        self, state: np.ndarray, action: int, reward: float,
        next_state: np.ndarray, done: bool
    ):
        """Store a transition in the replay buffer."""
        self.replay_buffer.push(Transition(state, action, reward, next_state, done))

    def train_step(self) -> Optional[float]:
        """
        Perform one training step with Double DQN.

        Returns:
            Loss value, or None if not enough samples
        """
        if len(self.replay_buffer) < BATCH_SIZE:
            return None

        transitions = self.replay_buffer.sample(BATCH_SIZE)

        states = torch.tensor(
            np.array([t.state for t in transitions]), dtype=torch.float32
        ).to(self.device)
        actions = torch.tensor(
            [t.action for t in transitions], dtype=torch.long
        ).to(self.device)
        rewards = torch.tensor(
            [t.reward for t in transitions], dtype=torch.float32
        ).to(self.device)
        next_states = torch.tensor(
            np.array([t.next_state for t in transitions]), dtype=torch.float32
        ).to(self.device)
        dones = torch.tensor(
            [t.done for t in transitions], dtype=torch.float32
        ).to(self.device)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + GAMMA * next_q * (1 - dones)

        loss = nn.functional.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self._soft_update_target()
        self._decay_epsilon()

        loss_val = loss.item()
        self.training_losses.append(loss_val)
        self.steps += 1

        return loss_val

    def _soft_update_target(self):
        """Soft update target network parameters: target = tau * policy + (1-tau) * target."""
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                TAU * policy_param.data + (1.0 - TAU) * target_param.data
            )

    def _decay_epsilon(self):
        """Linear epsilon decay from EPSILON_START to EPSILON_END."""
        self.epsilon = max(
            EPSILON_END,
            EPSILON_START - (EPSILON_START - EPSILON_END) * self.steps / EPSILON_DECAY_STEPS,
        )

    def train_on_history(
        self, trades: List[Dict], n_epochs: int = 10
    ) -> Dict:
        """
        Train on historical trade data.

        Args:
            trades: List of trade dicts, each containing:
                - state: [position, pnl, vol, signal, regime, dd, wr]
                - pnl: realized pnl
                - volatility: market volatility at trade time
                - drawdown: drawdown at trade time

        Returns:
            Training statistics
        """
        if not trades:
            raise ValueError("No trades provided for training")

        all_states = np.array([t["state"] for t in trades])
        self.state_mean = all_states.mean(axis=0)
        self.state_std = all_states.std(axis=0) + 1e-8

        for epoch in range(n_epochs):
            epoch_losses = []

            for i in range(len(trades) - 1):
                trade = trades[i]
                state = (np.array(trade["state"]) - self.state_mean) / self.state_std
                next_state = (
                    (np.array(trades[i + 1]["state"]) - self.state_mean) / self.state_std
                )

                action = self.select_action(state, training=True)
                reward = self.compute_reward(
                    trade["pnl"], trade["volatility"], trade["drawdown"]
                )
                done = i == len(trades) - 2

                self.store_transition(state, action, reward, next_state, done)
                loss = self.train_step()
                if loss is not None:
                    epoch_losses.append(loss)

            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            logger.info(
                f"Epoch {epoch + 1}/{n_epochs} - "
                f"Avg Loss: {avg_loss:.6f} - Epsilon: {self.epsilon:.4f}"
            )

        return {
            "epochs": n_epochs,
            "total_steps": self.steps,
            "final_epsilon": self.epsilon,
            "buffer_size": len(self.replay_buffer),
            "avg_loss": float(np.mean(self.training_losses[-100:])) if self.training_losses else 0.0,
        }

    def save(self, path: str):
        """Save model and training state."""
        torch.save(
            {
                "policy_state": self.policy_net.state_dict(),
                "target_state": self.target_net.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps": self.steps,
                "state_mean": self.state_mean,
                "state_std": self.state_std,
                "training_losses": self.training_losses[-1000:],
            },
            path,
        )
        logger.info(f"RL model saved to {path}")

    def load(self, path: str):
        """Load model and training state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint["policy_state"])
        self.target_net.load_state_dict(checkpoint["target_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]
        self.state_mean = checkpoint["state_mean"]
        self.state_std = checkpoint["state_std"]
        self.training_losses = checkpoint.get("training_losses", [])
        logger.info(f"RL model loaded from {path}")
