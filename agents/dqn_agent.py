import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from agents.q_network import QNetwork
from agents.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.95,
        buffer_size=10000,
        batch_size=64,
        device=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.batch_size = batch_size

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state, epsilon: float, action_mask=None):
        if random.random() < epsilon:
            valid_actions = self._valid_actions_from_mask(action_mask)
            if valid_actions:
                return int(random.choice(valid_actions))
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_net(state_tensor).squeeze(0)
            valid_actions = self._valid_actions_from_mask(action_mask)
            if valid_actions:
                mask_tensor = torch.tensor(
                    action_mask,
                    dtype=torch.float32,
                    device=self.device,
                )
                q_values = q_values.masked_fill(mask_tensor <= 0, -1e9)

        return int(q_values.argmax().item())

    def _valid_actions_from_mask(self, action_mask):
        if action_mask is None:
            return []
        return [
            action_id
            for action_id, is_valid in enumerate(action_mask)
            if is_valid > 0
        ]

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.q_net(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            next_q_value = next_q_values.max(dim=1)[0]
            target = rewards + self.gamma * next_q_value * (1.0 - dones)

        loss = F.mse_loss(q_value, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path, completed_episodes=None, epsilon=None):
        checkpoint = {
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "completed_episodes": completed_episodes,
            "epsilon": epsilon,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
        }
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        if "q_net" not in checkpoint:
            self.q_net.load_state_dict(checkpoint)
            self.target_net.load_state_dict(self.q_net.state_dict())
            return {}

        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint.get("target_net", checkpoint["q_net"]))
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint
