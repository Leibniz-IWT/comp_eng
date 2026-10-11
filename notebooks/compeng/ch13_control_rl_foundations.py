"""Helper code for chapter 13: reinforcement learning foundations.

The notebook keeps the RL concepts visible (MDP tables, Bellman backups, tabular
update rules, loss functions and the stable-baselines3 ``PPO(...)`` call). This
module holds the plotting code, the deep-RL training loops with their on-disk
caches in ``notebooks/cache/`` and the episode videos in ``notebooks/media/``.

Training runs only when the cache file is missing or ``recompute=True`` is passed;
otherwise the stored networks and training curves are loaded in well under a second.
"""

from __future__ import annotations

import copy
import json
import os
import random
import time
import warnings
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch, Rectangle
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from torch.distributions import Normal

NOTEBOOKS = Path(__file__).resolve().parents[1]
CACHE = NOTEBOOKS / "cache"
MEDIA = NOTEBOOKS / "media"

GRID_ARROWS = [(0, 0.3), (0, -0.3), (-0.3, 0), (0.3, 0)]  # up, down, left, right
CLIFF_ARROWS = [(0, 0.3), (0.3, 0), (0, -0.3), (-0.3, 0)]  # CliffWalking: up, right, down, left


# ---------------------------------------------------------------------------
# 13.5 Dynamic programming on the 3x3 gridworld
# ---------------------------------------------------------------------------


def _arrow(ax: Axes, x: float, y: float, dxy: tuple[float, float], scale: float = 20) -> None:
    """Draw a policy arrow centred at (x, y) pointing along ``dxy``."""
    dx, dy = dxy
    ax.add_patch(
        FancyArrowPatch(
            (x - dx / 2, y - dy / 2), (x + dx / 2, y + dy / 2),
            arrowstyle="->", mutation_scale=scale, color="tab:blue", lw=1.8,
        )
    )


def plot_gridworld(
    policy: np.ndarray, V: np.ndarray, terminals: Sequence[int] = (4, 8), title: str = "", ax: Axes | None = None
) -> Axes:
    """Draw the 3x3 gridworld with policy arrows and state values V(s) (goal green, hole red)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4.8, 4.8))
    for s in range(9):
        i, j = divmod(s, 3)
        x, y = j + 0.5, 2.5 - i  # cell centre, row 0 at the top
        color = {8: "lightgreen", 4: "lightcoral"}.get(s, "white")
        ax.add_patch(Rectangle((j, 2 - i), 1, 1, facecolor=color, edgecolor="0.6", alpha=0.6))
        if s in terminals:
            ax.text(x, y, f"V={V[s]:.2f}", ha="center", va="center", fontsize=11)
            continue
        _arrow(ax, x, y + 0.15, GRID_ARROWS[policy[s]])
        ax.text(x, y - 0.25, f"V={V[s]:.2f}", ha="center", va="center", fontsize=11)
    ax.set(xlim=(0, 3), ylim=(0, 3), aspect="equal", title=title)
    ax.set_axis_off()
    return ax


def plot_policy_iteration(history: Sequence[tuple[np.ndarray, np.ndarray]], terminals: Sequence[int] = (4, 8)) -> np.ndarray:
    """Show policy and values for the initial guess and after every policy-iteration step."""
    nrows = -(-len(history) // 3)
    fig, axes = plt.subplots(nrows, 3, figsize=(9, 3.3 * nrows))
    for k, ax in enumerate(axes.flat):
        if k < len(history):
            policy, V = history[k]
            plot_gridworld(policy, V, terminals, title="initial policy" if k == 0 else f"iteration {k - 1}", ax=ax)
        else:
            ax.set_axis_off()
    fig.suptitle("Policy iteration (goal: green, hole: red)")
    return axes


# ---------------------------------------------------------------------------
# 13.6 Learning curves and the CliffWalking policy
# ---------------------------------------------------------------------------


def plot_learning_curve(
    rewards: Sequence[float], title: str = "", xlabel: str = "Episode", ylabel: str = "Total Reward", ax: Axes | None = None
) -> Axes:
    """Plot the total reward of every training episode."""
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(np.arange(len(rewards)), rewards)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    return ax


def plot_mc_learning_curve(episode_returns: Sequence[float], block: int = 100, window: int = 5000) -> Axes:
    """Average return of MC control on Blackjack: per block of episodes and as a long moving average."""
    returns = np.asarray(episode_returns, dtype=float)
    n_blocks = len(returns) // block
    block_means = returns[: n_blocks * block].reshape(n_blocks, block).mean(axis=1)
    moving = np.convolve(returns, np.ones(window) / window, mode="valid")
    _, ax = plt.subplots()
    ax.plot((np.arange(n_blocks) + 1) * block, block_means, color="0.75", lw=1, label=f"average of {block} episodes")
    ax.plot(np.arange(window, len(returns) + 1), moving, color="tab:blue", lw=2.5,
            label=f"moving average of {window} episodes")
    ax.set(xlabel="Episode", ylabel="Average Return", title="MC Control on Blackjack")
    ax.legend(loc="lower right")
    return ax


def plot_cliff_policy(policy: np.ndarray, values: np.ndarray) -> Axes:
    """Draw the greedy CliffWalking policy and values max_a Q(s, a) on the 4x12 grid."""
    rows, cols, start, goal = 4, 12, 36, 47
    _, ax = plt.subplots(figsize=(12, 4.6))
    for s in range(rows * cols):
        i, j = divmod(s, cols)
        x, y = j + 0.5, rows - 0.5 - i  # cell centre, row 0 at the top
        cliff = 37 <= s <= 46
        color = "lightgreen" if s == start else "gold" if s == goal else "lightcoral" if cliff else "white"
        ax.add_patch(Rectangle((j, rows - 1 - i), 1, 1, facecolor=color, edgecolor="0.7", alpha=0.6))
        if s == goal or cliff:  # no policy on the goal and the cliff
            ax.text(x, y, "Goal" if s == goal else f"V={values[s]:.1f}", ha="center", va="center", fontsize=10)
            continue
        _arrow(ax, x, y + 0.12, CLIFF_ARROWS[policy[s]], scale=15)
        ax.text(x, y - 0.3, f"V={values[s]:.1f}", ha="center", va="center", fontsize=9)
    ax.set(
        xlim=(0, cols), ylim=(0, rows), aspect="equal",
        title="SARSA Learned Policy and Values on CliffWalking\n(Green: Start, Gold: Goal, Red: Cliff)",
    )
    ax.set_axis_off()
    return ax


# ---------------------------------------------------------------------------
# Cache helpers for the hand-written PyTorch agents
# ---------------------------------------------------------------------------


@dataclass
class Agent:
    """A policy (observation -> action) together with its training curves and network."""

    policy: Callable[[np.ndarray], Any]
    curves: dict[str, np.ndarray] = field(default_factory=dict)
    net: nn.Module | None = None


@contextmanager
def _torch_threads(n: int = 1) -> Iterator[None]:
    """Temporarily limit torch's CPU threads (tiny networks train fastest on one thread)."""
    old = torch.get_num_threads()
    torch.set_num_threads(n)
    try:
        yield
    finally:
        torch.set_num_threads(old)


def _save(path: Path, arrays: dict[str, Any], config: dict[str, Any], **nets: nn.Module) -> None:
    """Store training curves, network weights and the training settings in one compressed .npz file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = {f"{name}.{k}": v.detach().cpu().numpy() for name, net in nets.items() for k, v in net.state_dict().items()}
    np.savez_compressed(path, config=json.dumps(config), **{k: np.asarray(v) for k, v in arrays.items()}, **weights)


def _load(path: Path, config: dict[str, Any], **nets: nn.Module) -> dict[str, np.ndarray]:
    """Load the weights stored by :func:`_save` into ``nets``; return the other arrays."""
    with np.load(path) as data:
        for name, net in nets.items():
            prefix = f"{name}."
            net.load_state_dict({k[len(prefix):]: torch.from_numpy(data[k]) for k in data.files if k.startswith(prefix)})
        arrays = {k: data[k] for k in data.files if "." not in k and k != "config"}
        stored = json.loads(str(data["config"]))
    note = "" if stored == config else f" (trained with different settings: {stored})"
    print(f"Loaded {path.relative_to(NOTEBOOKS)}{note}; pass recompute=True to retrain")
    return arrays


def _report_training(path: Path, t0: float) -> None:
    """Print where a freshly trained agent was cached and how long training took."""
    print(f"Trained in {time.perf_counter() - t0:.0f} s, saved {path.relative_to(NOTEBOOKS)}")


# ---------------------------------------------------------------------------
# 13.7.2 DQN on CartPole
# ---------------------------------------------------------------------------


class DQN(nn.Module):
    """Fully connected Q-network: observation -> one Q-value per action."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Q(s, a) for all actions a."""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


def _greedy(net: nn.Module) -> Callable[[np.ndarray], int]:
    """Greedy policy argmax_a Q(s, a) of a Q-network."""

    def policy(obs: np.ndarray) -> int:
        with torch.no_grad():
            return int(net(torch.as_tensor(obs, dtype=torch.float32)).argmax())

    return policy


def train_or_load_dqn(
    loss_fn: Callable[..., torch.Tensor],
    hidden_dim: int = 64,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    batch_size: int = 64,
    lr: float = 1e-3,
    target_update: int = 10,
    buffer_size: int = 10_000,
    num_episodes: int = 500,
    seed: int = 0,
    recompute: bool = False,
) -> Agent:
    """Train a DQN on CartPole-v1 with ``loss_fn(policy_net, target_net, batch, gamma)``, or load it from the cache."""
    config = {k: v for k, v in locals().items() if k not in ("loss_fn", "recompute")}
    path = CACHE / "ch13_dqn_cartpole.npz"
    env = gym.make("CartPole-v1")
    torch.manual_seed(seed)
    policy_net = DQN(env.observation_space.shape[0], env.action_space.n, hidden_dim)
    if path.exists() and not recompute:
        return Agent(_greedy(policy_net), _load(path, config, policy_net=policy_net), policy_net)

    t0 = time.perf_counter()
    rng = random.Random(seed)
    env.action_space.seed(seed)
    target_net = copy.deepcopy(policy_net)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer: deque[tuple[Any, ...]] = deque(maxlen=buffer_size)
    episode_rewards, epsilon = [], epsilon_start
    with _torch_threads(1):
        for episode in range(num_episodes):
            state, _ = env.reset(seed=seed if episode == 0 else None)
            total_reward, done = 0.0, False
            while not done:
                if rng.random() < epsilon:  # epsilon-greedy exploration
                    action = int(env.action_space.sample())
                else:
                    with torch.no_grad():
                        action = int(policy_net(torch.as_tensor(state, dtype=torch.float32)).argmax())
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated  # CartPole-v1 episodes are truncated after 500 steps
                total_reward += reward
                replay_buffer.append((state, action, reward, next_state, terminated))
                state = next_state
                if len(replay_buffer) > batch_size:  # learn from a random minibatch of the replay buffer
                    states, actions, rewards, next_states, dones = zip(*rng.sample(replay_buffer, batch_size), strict=True)
                    batch = (
                        torch.as_tensor(np.array(states), dtype=torch.float32),
                        torch.as_tensor(actions).unsqueeze(1),
                        torch.as_tensor(rewards, dtype=torch.float32),
                        torch.as_tensor(np.array(next_states), dtype=torch.float32),
                        torch.as_tensor(dones, dtype=torch.float32),
                    )
                    loss = loss_fn(policy_net, target_net, batch, gamma)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            episode_rewards.append(total_reward)
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            if episode % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())
    curves = {"episode_rewards": np.array(episode_rewards)}
    _save(path, curves, config, policy_net=policy_net)
    _report_training(path, t0)
    return Agent(_greedy(policy_net), curves, policy_net)


# ---------------------------------------------------------------------------
# 13.8.2 REINFORCE with baseline on Pendulum
# ---------------------------------------------------------------------------


class PolicyNet(nn.Module):
    """Gaussian policy pi(u|s): mean 2 tanh(.) inside the torque limits, softplus standard deviation."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_sigma = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean and standard deviation of the action distribution."""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = 2 * torch.tanh(self.fc_mu(x))  # Scale to [-2, 2]
        sigma = F.softplus(self.fc_sigma(x)) + 1e-5
        return mu, sigma


class ValueNet(nn.Module):
    """State-value network V(s), used as the REINFORCE baseline."""

    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return V(s)."""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def train_or_load_reinforce(
    loss_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    hidden_dim: int = 128,
    lr_actor: float = 3e-4,
    lr_critic: float = 1e-3,
    gamma: float = 0.95,
    num_episodes: int = 3000,
    max_steps: int = 300,
    entropy_beta: float = 0.01,
    clip_grad_norm: float = 1.0,
    num_envs: int = 4,
    seed: int = 42,
    recompute: bool = False,
) -> Agent:
    """Train REINFORCE with baseline on Pendulum-v1 with ``loss_fn``, or load it from the cache.

    Each of the ``num_episodes`` updates uses ``num_envs`` episodes of ``max_steps`` steps, run side
    by side; ``loss_fn(log_probs, entropies, values, returns, entropy_beta)`` gets one entry per time
    step of these episodes and returns (actor loss, critic loss).
    """
    config = {k: v for k, v in locals().items() if k not in ("loss_fn", "recompute")}
    path = CACHE / "ch13_reinforce_pendulum.npz"
    torch.manual_seed(seed)
    policy_net, value_net = PolicyNet(3, 1, hidden_dim), ValueNet(3, hidden_dim)

    def policy(obs: np.ndarray) -> np.ndarray:  # mean action of the Gaussian policy
        with torch.no_grad():
            return policy_net(torch.as_tensor(obs, dtype=torch.float32))[0].numpy()

    if path.exists() and not recompute:
        return Agent(policy, _load(path, config, policy_net=policy_net, value_net=value_net), policy_net)

    t0 = time.perf_counter()
    envs = [gym.make("Pendulum-v1", max_episode_steps=max_steps) for _ in range(num_envs)]
    for i, env in enumerate(envs):
        env.reset(seed=seed + i)
    actor_optim = torch.optim.AdamW(policy_net.parameters(), lr=lr_actor)
    critic_optim = torch.optim.AdamW(value_net.parameters(), lr=lr_critic)
    curves: dict[str, list[float]] = {k: [] for k in ("episode_rewards", "avg_entropies", "actor_losses", "critic_losses")}
    with _torch_threads(1):
        for _ in range(num_episodes):
            # several trajectories per update reduce the gradient variance; they run side by side
            obs = np.array([env.reset()[0] for env in envs], dtype=np.float32)
            states, actions, rewards = [], [], []
            for _ in range(max_steps):  # Pendulum never terminates: every episode lasts max_steps steps
                with torch.no_grad():
                    mu, sigma = policy_net(torch.as_tensor(obs))
                    action = Normal(mu, sigma).sample()
                states.append(obs)
                actions.append(action)
                steps = [env.step(action[i].numpy()) for i, env in enumerate(envs)]
                obs = np.array([s[0] for s in steps], dtype=np.float32)
                rewards.append([float(s[1]) for s in steps])
            R = np.array(rewards)  # (time step, episode)
            G, g = np.zeros_like(R), np.zeros(num_envs)
            for t in reversed(range(max_steps)):  # discounted return G_t (reward-to-go) of every episode
                g = R[t] + gamma * g
                G[t] = g
            # log-probabilities, entropies and baselines V(s_t) of all time steps in one pass (same network)
            S = torch.as_tensor(np.array(states)).reshape(max_steps * num_envs, 3)
            mu, sigma = policy_net(S)
            dist = Normal(mu, sigma)
            log_probs = dist.log_prob(torch.stack(actions).reshape(max_steps * num_envs, 1)).squeeze(-1)
            entropies, values = dist.entropy().squeeze(-1), value_net(S).squeeze(-1)
            returns = torch.as_tensor(G.reshape(max_steps * num_envs), dtype=torch.float32)
            actor_loss, critic_loss = loss_fn(log_probs, entropies, values, returns, entropy_beta)
            actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), clip_grad_norm)
            actor_optim.step()
            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), clip_grad_norm)
            critic_optim.step()
            curves["episode_rewards"].append(float(R.sum(axis=0).mean()))  # average return per episode
            curves["avg_entropies"].append(entropies.mean().item())
            curves["actor_losses"].append(actor_loss.item())
            curves["critic_losses"].append(critic_loss.item())
    arrays = {k: np.array(v) for k, v in curves.items()}
    _save(path, arrays, config, policy_net=policy_net, value_net=value_net)
    _report_training(path, t0)
    return Agent(policy, arrays, policy_net)


def plot_reinforce_training(curves: dict[str, np.ndarray]) -> np.ndarray:
    """Episode rewards, policy entropy and the two losses of the REINFORCE training run."""
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    panels = [
        ("episode_rewards", "Episode Rewards"),
        ("avg_entropies", "Average Entropy"),
        ("actor_losses", "Actor Losses"),
        ("critic_losses", "Critic Losses"),
    ]
    for ax, (key, title) in zip(axs.flat, panels, strict=True):
        ax.plot(curves[key], lw=0.8)
        ax.set_title(title)
    for ax in axs[1]:
        ax.set_xlabel("Episode (update)")
    fig.suptitle("REINFORCE with baseline on Pendulum")
    return axs


# ---------------------------------------------------------------------------
# 13.8.2 stable-baselines3 PPO with an on-disk cache
# ---------------------------------------------------------------------------


class EpisodeLogger(BaseCallback):
    """Record the timestep and return of every finished training episode (from the Monitor wrapper)."""

    def __init__(self) -> None:
        super().__init__()
        self.timesteps: list[int] = []
        self.returns: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" in info:
                self.timesteps.append(self.num_timesteps)
                self.returns.append(float(info["episode"]["r"]))
        return True


def train_or_load(
    model: BaseAlgorithm, total_timesteps: int, name: str, recompute: bool = False
) -> tuple[BaseAlgorithm, dict[str, np.ndarray]]:
    """Run ``model.learn(total_timesteps)`` once and cache it as cache/ch13_<name>.zip; later calls load the cache.

    Returns the (trained or loaded) model and its training curve (timesteps, episode returns).
    """
    path = CACHE / f"ch13_{name}.zip"
    curve = path.with_suffix(".npz")
    if path.exists() and curve.exists() and not recompute:
        model = type(model).load(path, env=model.get_env(), device=model.device)
        print(f"Loaded {path.relative_to(NOTEBOOKS)} ({model.num_timesteps} timesteps); pass recompute=True to retrain")
    else:
        logger = EpisodeLogger()
        t0 = time.perf_counter()
        with _torch_threads(1):
            model.learn(total_timesteps=total_timesteps, callback=logger)
        CACHE.mkdir(parents=True, exist_ok=True)
        model.save(path)
        np.savez_compressed(curve, timesteps=np.array(logger.timesteps), returns=np.array(logger.returns))
        _report_training(path, t0)
    with np.load(curve) as data:
        return model, {k: data[k] for k in data.files}


def plot_training_curve(log: dict[str, np.ndarray], title: str = "", window: int = 40, ax: Axes | None = None) -> Axes:
    """Episode returns during training versus environment steps, with a moving average."""
    t, r = log["timesteps"], log["returns"]
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(t, r, ".", color="0.7", ms=3, label="episode return")
    ax.plot(t[window - 1 :], np.convolve(r, np.ones(window) / window, mode="valid"), lw=2.5,
            label=f"moving average of {window} episodes")
    ax.set(xlabel="Environment steps", ylabel="Episode Return", title=title)
    ax.legend(loc="lower right")
    return ax


# ---------------------------------------------------------------------------
# 13.8.3 A2C on MountainCar
# ---------------------------------------------------------------------------


class RunningNorm:
    """Running mean and variance of the observations, used to normalize the network input."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = 1e-4

    def update(self, batch: np.ndarray) -> None:
        """Merge the statistics of ``batch`` (rows are observations)."""
        batch = np.asarray(batch, dtype="float64")
        batch_mean, batch_var, batch_count = batch.mean(axis=0), batch.var(axis=0), batch.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        m2 = self.var * self.count + batch_var * batch_count + np.square(delta) * self.count * batch_count / tot_count
        self.var = m2 / tot_count
        self.mean = self.mean + delta * batch_count / tot_count
        self.count = tot_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Return the normalized observation."""
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


class ActorCritic(nn.Module):
    """Shared trunk with a policy head (action logits) and a value head V(s)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the action logits and V(s)."""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.actor(x), self.critic(x)


def _a2c_policy(net: ActorCritic, norm: RunningNorm, greedy: bool, seed: int = 0) -> Callable[[np.ndarray], int]:
    """Greedy (argmax) or sampling policy of an actor-critic network with its observation normalizer."""
    generator = torch.Generator().manual_seed(seed)

    def policy(obs: np.ndarray) -> int:
        with torch.no_grad():
            logits, _ = net(torch.as_tensor(norm.normalize(obs), dtype=torch.float32))
        if greedy:
            return int(torch.argmax(logits))
        return int(torch.multinomial(torch.softmax(logits, dim=0), 1, generator=generator))

    return policy


def train_or_load_a2c(
    loss_fn: Callable[..., torch.Tensor],
    hidden_dim: int = 256,
    lr: float = 7e-4,
    gamma: float = 0.99,
    lam: float = 0.95,
    num_episodes: int = 8000,
    max_steps: int = 200,
    velocity_bonus: float = 10.0,
    seed: int = 0,
    recompute: bool = False,
) -> tuple[Agent, Agent]:
    """Train A2C on MountainCar-v0 with ``loss_fn``, or load it; return the (untrained, trained) agents.

    The untrained agent samples from the freshly initialised network, the trained one acts greedily.
    ``loss_fn(log_probs, entropies, values, returns, advantages)`` returns the total loss of one episode.
    """
    config = {k: v for k, v in locals().items() if k not in ("loss_fn", "recompute")}
    path = CACHE / "ch13_a2c_mountaincar.npz"
    torch.manual_seed(seed)
    net, norm = ActorCritic(2, 3, hidden_dim), RunningNorm((2,))
    before = Agent(_a2c_policy(copy.deepcopy(net), copy.deepcopy(norm), greedy=False, seed=seed))
    if path.exists() and not recompute:
        arrays = _load(path, config, net=net)
        norm.mean, norm.var, norm.count = arrays.pop("norm_mean"), arrays.pop("norm_var"), float(arrays.pop("norm_count"))
        return before, Agent(_a2c_policy(net, norm, greedy=True), arrays, net)

    t0 = time.perf_counter()
    env = gym.make("MountainCar-v0")
    env.reset(seed=seed)
    optimizer = torch.optim.RMSprop(net.parameters(), lr=lr)
    episode_rewards = []
    with _torch_threads(1):
        for _ in range(num_episodes):
            state, _ = env.reset()
            states, log_probs, values, rewards, entropies = [state.copy()], [], [], [], []
            done, step = False, 0
            while not done and step < max_steps:
                policy_logits, value = net(torch.as_tensor(norm.normalize(state), dtype=torch.float32))
                dist = torch.softmax(policy_logits, dim=0)
                action = torch.multinomial(dist, 1).item()
                log_probs.append(torch.log(dist[action]))
                entropies.append(-torch.sum(dist * torch.log(dist + 1e-10)))
                next_state, reward, done, _, _ = env.step(action)
                reward += velocity_bonus * abs(next_state[1])  # reward shaping: bonus for speed
                values.append(value)
                rewards.append(float(reward))
                states.append(next_state.copy())
                state = next_state
                step += 1
            norm.update(np.array(states))
            with torch.no_grad():
                _, next_value = net(torch.as_tensor(norm.normalize(state), dtype=torch.float32))
            values_t = torch.cat(values)
            v = values_t.detach().numpy()
            advantages, gae = [], 0.0
            for t in reversed(range(len(rewards))):  # generalized advantage estimation (GAE)
                last = t == len(rewards) - 1
                next_non_terminal = 1.0 - done if last else 1.0
                next_v = next_value.item() if last else v[t + 1]
                delta = rewards[t] + gamma * next_v * next_non_terminal - v[t]
                gae = delta + gamma * lam * next_non_terminal * gae
                advantages.insert(0, gae)
            advantages_t = torch.tensor(advantages, dtype=torch.float32)
            returns = advantages_t + values_t.detach()  # critic target, no gradient through it
            loss = loss_fn(torch.stack(log_probs), torch.stack(entropies), values_t, returns, advantages_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            episode_rewards.append(sum(rewards))
    arrays = {"episode_rewards": np.array(episode_rewards)}
    _save(path, arrays | {"norm_mean": norm.mean, "norm_var": norm.var, "norm_count": norm.count}, config, net=net)
    _report_training(path, t0)
    return before, Agent(_a2c_policy(net, norm, greedy=True), arrays, net)


def plot_mountaincar_position(policy: Callable[[np.ndarray], int], max_steps: int = 400, seed: int = 0) -> Axes:
    """Roll out ``policy`` on MountainCar-v0 and plot the car position over the steps."""
    env = gym.make("MountainCar-v0", max_episode_steps=max_steps)
    state, _ = env.reset(seed=seed)
    positions, done = [state[0]], False
    while not done:
        state, _, terminated, truncated, _ = env.step(policy(state))
        positions.append(state[0])
        done = terminated or truncated
    _, ax = plt.subplots()
    ax.plot(positions, label="car position")
    ax.axhline(0.5, color="tab:red", label="goal (flag)")
    ax.set(xlabel="Step", ylabel="Position", title="A2C on MountainCar: one episode after training")
    ax.legend(loc="lower right")
    return ax


# ---------------------------------------------------------------------------
# Episode videos (rendered once, embedded from markdown)
# ---------------------------------------------------------------------------


def _save_animation(frames: Sequence[np.ndarray], path: Path, fps: int, title: str) -> Path:
    """Write RGB frames as H.264 mp4 with ffmpeg; without ffmpeg fall back to a gif next to ``path``."""
    h, w = frames[0].shape[:2]
    top = 40 if title else 0
    # manual axes placement: savefig would re-enable the course default (constrained layout) between frames
    with plt.rc_context({"figure.constrained_layout.use": False}):
        fig = plt.figure(figsize=(w / 100, (h + top) / 100), dpi=100)
        ax = fig.add_axes((0, 0, 1, h / (h + top)))
        ax.set_axis_off()
        img = ax.imshow(frames[0])
        if title:
            fig.text(0.5, 1 - top / 2 / (h + top), title, ha="center", va="center", fontsize=15)

        def update(i: int) -> list[Any]:
            img.set_data(frames[i])
            return [img]

        anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps)
        if animation.writers.is_available("ffmpeg"):
            writer: animation.AbstractMovieWriter = animation.FFMpegWriter(
                fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "28", "-movflags", "+faststart"]
            )
        else:
            path, writer = path.with_suffix(".gif"), animation.PillowWriter(fps=fps)
        path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(path, writer=writer, dpi=100)
        plt.close(fig)
    return path


def record_video(
    policy: BaseAlgorithm | Callable[[np.ndarray], Any],
    env_id: str,
    path: str,
    max_steps: int = 200,
    title: str = "",
    fps: int = 20,
    seed: int = 0,
    recompute: bool = False,
) -> None:
    """Roll out ``policy`` (SB3 model or callable obs -> action) once and save the episode as notebooks/``path``.

    The video is rendered only if the file is missing or ``recompute=True``; embed it from a markdown cell.
    """
    target = NOTEBOOKS / path
    if target.exists() and not recompute:
        print(f"{path}: cached ({target.stat().st_size / 1e6:.2f} MB)")
        return
    act = (lambda obs: policy.predict(obs, deterministic=True)[0]) if isinstance(policy, BaseAlgorithm) else policy
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    with warnings.catch_warnings():  # pygame import noise (rendering only)
        warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated")
        warnings.filterwarnings("ignore", message=".*avx2 capable")
        env = gym.make(env_id, render_mode="rgb_array", max_episode_steps=max_steps)
        obs, _ = env.reset(seed=seed)
        frames, done = [env.render()], False
        while not done:
            obs, _, terminated, truncated, _ = env.step(act(obs))
            frames.append(env.render())
            done = terminated or truncated
        env.close()
    target = _save_animation(frames, target, fps, title)
    print(f"{target.relative_to(NOTEBOOKS)}: {len(frames)} frames written ({target.stat().st_size / 1e6:.2f} MB)")


# ---------------------------------------------------------------------------
# 13.9 Exploration (bandit) and reward shaping (toy GridWorld)
# ---------------------------------------------------------------------------


def plot_regret(regrets: dict[str, Sequence[float]]) -> Axes:
    """Cumulative regret of bandit strategies over the number of pulls."""
    _, ax = plt.subplots()
    for label, regret in regrets.items():
        ax.plot(np.arange(1, len(regret) + 1), regret, label=label)
    ax.set(xlabel="Pulls", ylabel="Cumulative Regret", title="Exploration Strategies in Multi-Armed Bandit")
    ax.legend()
    return ax


class GridWorld:
    """Toy 5x5 grid: start (0, 0), goal (4, 4), actions up/down/left/right, reward +1 at the goal, -1 per other step."""

    def __init__(self, size: int = 5) -> None:
        self.size = size
        self.start = (0, 0)
        self.goal = (4, 4)

    def step(self, state: tuple[int, int], action: int) -> tuple[tuple[int, int], int, bool]:
        """Apply ``action`` (0 up, 1 down, 2 left, 3 right, clipped at the walls)."""
        x, y = state
        if action == 0:
            y = min(y + 1, self.size - 1)  # up
        elif action == 1:
            y = max(y - 1, 0)  # down
        elif action == 2:
            x = max(x - 1, 0)  # left
        elif action == 3:
            x = min(x + 1, self.size - 1)  # right
        next_state = (x, y)
        reward = 1 if next_state == self.goal else -1
        return next_state, reward, next_state == self.goal


def q_learning_gridworld(
    env: GridWorld,
    shaping: Callable[[tuple[int, int], tuple[int, int], tuple[int, int], float], float] | None = None,
    episodes: int = 500,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    seed: int = 0,
) -> tuple[list[float], np.ndarray]:
    """Tabular Q-learning on ``env``; ``shaping(state, next_state, goal, gamma)`` is added to the learning reward.

    Returns the environment return of every episode (without the shaping term, so that
    runs with and without shaping are comparable) and the Q-table Q[x, y, action].
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.size, env.size, 4))
    returns = []
    for _ in range(episodes):
        state, total_reward, done = env.start, 0, False
        while not done:
            action = int(rng.integers(4)) if rng.random() < epsilon else int(np.argmax(Q[state]))
            next_state, reward, done = env.step(state, action)
            total_reward += reward
            if shaping is not None:
                reward += shaping(state, next_state, env.goal, gamma)
            Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])
            state = next_state
        returns.append(total_reward)
    return returns, Q


def plot_shaping_rewards(rewards_base: Sequence[float], rewards_shaped: Sequence[float]) -> Axes:
    """Learning curves of Q-learning without and with reward shaping (environment reward)."""
    _, ax = plt.subplots()
    ax.plot(rewards_base, label="Baseline")
    ax.plot(rewards_shaped, label="With Shaping")
    ax.set(xlabel="Episode", ylabel="Total Reward (without shaping term)", title="Reward Shaping Demo on Toy GridWorld")
    ax.legend(loc="lower right")
    return ax


def plot_gridworld_q(Q: np.ndarray, title: str, goal: tuple[int, int] = (4, 4)) -> np.ndarray:
    """Greedy policy arrows and value heatmap max_a Q(s, a) of the toy GridWorld."""
    size = Q.shape[0]
    policy, values = np.argmax(Q, axis=2), np.max(Q, axis=2)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.6))
    ax = axs[0]
    for x in range(size):
        for y in range(size):
            if (x, y) == goal:
                ax.text(x + 0.5, y + 0.5, "Goal", ha="center", va="center")
                continue
            _arrow(ax, x + 0.5, y + 0.5, GRID_ARROWS[policy[x, y]], scale=15)
    ax.set(xlim=(0, size), ylim=(0, size), aspect="equal", xticks=range(size + 1), yticks=range(size + 1),
           xlabel="X", ylabel="Y", title=f"Policy: {title}")
    im = axs[1].imshow(values.T, cmap="viridis", origin="lower")
    fig.colorbar(im, ax=axs[1], label=r"$\max_a Q(s,a)$")
    axs[1].set(xlabel="X", ylabel="Y", title=f"Values: {title}")
    axs[1].grid(False)
    return axs
