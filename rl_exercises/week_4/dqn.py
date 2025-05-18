"""
Deep Q-Learning implementation.
"""

from typing import Any, Dict, List, Tuple

import gymnasium as gym
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from rl_exercises.agent import AbstractAgent
from rl_exercises.week_4.buffers import ReplayBuffer
from rl_exercises.week_4.networks import QNetwork


def set_seed(env: gym.Env, seed: int = 0) -> None:
    """
    Seed Python, NumPy, PyTorch and the Gym environment for reproducibility.

    Parameters
    ----------
    env : gym.Env
        The Gym environment to seed.
    seed : int
        Random seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    # some spaces also support .seed()
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)


class DQNAgent(AbstractAgent):
    """
    Deep Q-Learning agent with ε-greedy policy and target network.

    Derives from AbstractAgent by implementing:
      - predict_action
      - save / load
      - update_agent
    """

    def __init__(
        self,
        env: gym.Env,
        buffer_capacity: int = 10000,  # wird von hydra überschrieben
        batch_size: int = 32,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.01,
        epsilon_decay: int = 500,
        target_update_freq: int = 1000,
        seed: int = 10,
        width=64,
        depth=4,
    ) -> None:
        """
        Initialize replay buffer, Q-networks, optimizer, and hyperparameters.

        Parameters
        ----------
        env : gym.Env
            The Gym environment.
        buffer_capacity : int
            Max experiences stored.
        batch_size : int
            Mini-batch size for updates.
        lr : float
            Learning rate.
        gamma : float
            Discount factor.
        epsilon_start : float
            Initial ε for exploration.
        epsilon_final : float
            Final ε.
        epsilon_decay : int
            Exponential decay parameter.
        target_update_freq : int
            How many updates between target-network syncs.
        seed : int
            RNG seed.
        """
        super().__init__(
            env,
            buffer_capacity,
            batch_size,
            lr,
            gamma,
            epsilon_start,
            epsilon_final,
            epsilon_decay,
            target_update_freq,
            seed,
            depth,
            width,
        )
        self.env = env
        set_seed(env, seed)

        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n

        # main Q-network and frozen target

        self.q = QNetwork(obs_dim, n_actions, depth=depth, hidden_dim=width)
        self.target_q = QNetwork(obs_dim, n_actions, depth=depth, hidden_dim=width)
        self.target_q.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)
        print(f"test:{buffer_capacity}")
        self.buffer = ReplayBuffer(buffer_capacity)

        # hyperparams
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq

        self.total_steps = 0  # for ε decay and target sync

        self.depth = depth
        self.width = width

    def epsilon(self) -> float:
        """
        Compute current ε by exponential decay.

        Returns
        -------
        float
            Exploration rate.
        """
        # TODO: implement exponential‐decayin
        # ε = ε_final + (ε_start - ε_final) * exp(-total_steps / ε_decay)
        # Currently, it is constant and returns the starting value ε

        return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * np.exp(
            -self.total_steps / self.epsilon_decay
        )

    def predict_action(
        self, state: np.ndarray, evaluate: bool = False
    ) -> Tuple[int, Dict]:
        """
        Choose action via ε-greedy (or purely greedy in eval mode).

        Parameters
        ----------
        state : np.ndarray
            Current observation.
        info : dict
            Gym info dict (unused here).
        evaluate : bool
            If True, always pick argmax(Q).

        Returns
        -------
        action : int
        info_out : dict
            Empty dict (compatible with interface).
        """

        stateTensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        if evaluate:
            # select purely greedy action from Q(s)
            with torch.no_grad():
                qvals = self.q.forward(stateTensor)
                action = qvals.argmax().item()

        else:
            qvals = self.q.forward(stateTensor)
            if np.random.rand() < self.epsilon():
                # sample random action
                action = np.random.randint(qvals.shape[1])

            else:
                # select purely greedy action from Q(s)

                action = qvals.argmax().item()

        return action

    def save(self, path: str) -> None:
        """
        Save model & optimizer state to disk.

        Parameters
        ----------
        path : str
            File path.
        """
        torch.save(
            {
                "parameters": self.q.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """
        Load model & optimizer state from disk.

        Parameters
        ----------
        path : str
            File path.
        """
        checkpoint = torch.load(path)
        self.q.load_state_dict(checkpoint["parameters"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def update_agent(
        self, training_batch: List[Tuple[Any, Any, float, Any, bool, Dict]]
    ) -> float:
        """
        Perform one gradient update on a batch of transitions.

        Parameters
        ----------
        training_batch : list of transitions
            Each is (state, action, reward, next_state, done, info).

        Returns
        -------
        loss_val : float
            MSE loss value.
        """
        # unpack
        states, actions, rewards, next_states, dones, _ = zip(*training_batch)  # noqa: F841
        s = torch.tensor(np.array(states), dtype=torch.float32)  # noqa: F841
        a = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(1)  # noqa: F841
        r = torch.tensor(np.array(rewards), dtype=torch.float32)  # noqa: F841
        s_next = torch.tensor(np.array(next_states), dtype=torch.float32)  # noqa: F841
        mask = torch.tensor(np.array(dones), dtype=torch.float32)  # noqa: F841

        # # TODO: pass batched states through self.q and gather Q(s,a)
        predQs = self.q.forward(s)
        pred = predQs.gather(1, a).squeeze(1)

        # TODO: compute TD target with frozen network
        with torch.no_grad():
            targetQs = self.target_q.forward(s_next)
            maxTargetQs = torch.max(targetQs, 1).values
            target = r + self.gamma * (1 - mask) * maxTargetQs

        loss = nn.MSELoss()(pred, target)

        # gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # occasionally sync target network
        if self.total_steps % self.target_update_freq == 0:
            self.target_q.load_state_dict(self.q.state_dict())

        self.total_steps += 1
        return float(loss.item())

    def train(self, num_frames: int, eval_interval: int = 1000) -> None:
        """
        Run a training loop for a fixed number of frames.

        Parameters
        ----------
        num_frames : int
            Total environment steps.
        eval_interval : int
            Every this many episodes, print average reward.
        """
        state, _ = self.env.reset()
        ep_reward = 0.0
        recent_rewards: List[float] = []
        corresponding_frame = []

        for frame in range(1, num_frames + 1):
            action = self.predict_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)

            # store and step
            self.buffer.add(state, action, reward, next_state, done or truncated, {})
            state = next_state
            ep_reward += reward

            # update if ready
            if len(self.buffer) >= self.batch_size:
                # TODO: sample a batch from replay buffer
                batch = self.buffer.sample(self.batch_size)
                _ = self.update_agent(batch)

            if done or truncated:
                state, _ = self.env.reset()
                recent_rewards.append(ep_reward)
                corresponding_frame.append(frame)
                ep_reward = 0.0
                # logging
                if len(recent_rewards) % 10 == 0:
                    # TODO: compute avg over last eval_interval episodes and print
                    avg = np.average(recent_rewards[-10:])
                    print(
                        f"Frame {frame}, AvgReward(10): {avg:.2f}, ε={self.epsilon():.3f}"
                    )
        self.plot_training_curve(recent_rewards, corresponding_frame)
        print("Training complete.")

    def plot_training_curve(self, rewards, corrFrame):
        cut = len(rewards) % 10
        rewards = rewards[cut:] if cut != 0 else rewards
        corrFrame = corrFrame[cut:] if cut != 0 else corrFrame
        meanRewards = []
        frames = []
        for i in range(int(len(rewards) / 10)):
            meanRewards.append(np.average(rewards[10 * i : 10 * i + 10]))
            frames.append(corrFrame[10 * i + 5])
        plt.figure(figsize=(10, 6))
        plt.plot(frames, meanRewards, label="Mean Reward")
        plt.xlabel("Frames")
        plt.ylabel("Mean Reward")
        plt.title(
            f"Training Curve - {self.__class__.__name__}\n Depth: {self.depth} | Width: {self.width} | Buffer-size: {self.buffer.capacity} | Batch-size: {self.batch_size}"
        )
        plt.grid(True)
        plt.legend()
        print(
            f"../../../rl_exercises/week_4/plots/th/training_curve_{self.__class__.__name__}_depth_{self.depth}_width_{self.width}_buffer_{self.buffer.capacity}_batch_{self.batch_size}.png"
        )
        ## muss auskommentiert werden da sonst tests nicht durchlaufen anderer arbeitsordner
        # plt.savefig(
        #     f"../../../rl_exercises/week_4/plots/th/training_curve_{self.__class__.__name__}_depth_{self.depth}_width_{self.width}_buffer_{self.buffer.capacity}_batch_{self.batch_size}.png"
        # )
        plt.show()
        import os

        save_path = os.path.abspath(
            os.path.join("..", "..", "..", "dqn_trained_model.pth")
        )
        torch.save(
            {
                "parameters": self.q.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            save_path,
        )
        print(f"Modell gespeichert unter: {save_path}")
        return


@hydra.main(config_path="../configs/agent/", config_name="dqn", version_base="1.1")
def main(cfg: DictConfig):
    # 1) build env
    # env = gym.make(cfg.env.name, render_mode="human")
    env = gym.make(cfg.env.name)

    set_seed(env, cfg.seed)

    # 3) TODO: instantiate & train the agent
    agent = DQNAgent(
        env=env,
        seed=cfg.seed,
        buffer_capacity=cfg.agent.buffer_capacity,
        depth=cfg.network.depth,
        width=cfg.network.width,
        batch_size=cfg.agent.batch_size,
    )
    print(f"Buffer size: {cfg.agent.buffer_capacity}")
    agent.train(cfg.train.num_frames, cfg.train.eval_interval)

    # 4) TODO: evaluate the agent


if __name__ == "__main__":
    main()
