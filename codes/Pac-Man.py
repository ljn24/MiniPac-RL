import argparse
from collections import defaultdict
import os
from pathlib import Path
import time
import tkinter as tk

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk
import yaml


UNIT = 100
Map_H = 5
Map_W = 5

START_POS = 0
GOAL_POS = 24
BEAN_POSITIONS = [(0, 2), (2, 3)]
GHOST_POSITIONS = {(1, 2), (2, 1), (3, 3)}
ACTION_SPACE = ['u', 'd', 'l', 'r']
ACTION_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ['up', 'down', 'left', 'right']
ACTION_ARROWS = ['^', 'v', '<', '>']

STEP_REWARD = -1
WALL_REWARD = -10
BEAN_REWARD = 10
GHOST_REWARD = -100
FAIL_REWARD = -100
SUCCESS_REWARD = 100

GAMMA = 0.95
ALL_BEANS_MASK = (1 << len(BEAN_POSITIONS)) - 1
ASSET_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ASSET_DIR / "config.yaml"

map_state = np.array([
    [ 0,  1,  2,  3,  4],
    [ 5,  6,  7,  8,  9],
    [10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19],
    [20, 21, 22, 23, 24]
])


def grid_to_canvas(col, row):
    origin = np.array([UNIT / 2, UNIT / 2])
    return origin[0] + col * UNIT, origin[1] + row * UNIT


def pos_to_grid(pos):
    return divmod(pos, Map_W)


def grid_to_pos(row, col):
    return row * Map_W + col


def is_terminal_state(state):
    pos, bean_mask = state
    row, col = pos_to_grid(pos)
    return (row, col) in GHOST_POSITIONS or pos == GOAL_POS


def all_states():
    for bean_mask in range(ALL_BEANS_MASK + 1):
        for pos in range(Map_H * Map_W):
            yield pos, bean_mask


def transition(state, action):
    """Deterministic model used by value iteration and sampled by MC."""
    pos, bean_mask = state
    row, col = pos_to_grid(pos)
    dr, dc = ACTION_DELTAS[action]
    nr, nc = row + dr, col + dc

    if not (0 <= nr < Map_H and 0 <= nc < Map_W):
        return (pos, bean_mask), WALL_REWARD, False

    next_pos = grid_to_pos(nr, nc)
    reward = STEP_REWARD
    next_mask = bean_mask

    for i, bean in enumerate(BEAN_POSITIONS):
        if (nr, nc) == bean and not (bean_mask & (1 << i)):
            next_mask |= 1 << i
            reward += BEAN_REWARD

    if (nr, nc) in GHOST_POSITIONS:
        return (next_pos, next_mask), GHOST_REWARD, True

    if next_pos == GOAL_POS:
        reward = SUCCESS_REWARD if next_mask == ALL_BEANS_MASK else FAIL_REWARD
        return (next_pos, next_mask), reward, True

    return (next_pos, next_mask), reward, False


class Map(tk.Tk, object):
    def __init__(self):
        super(Map, self).__init__()
        self.action_space = ACTION_SPACE
        self.n_actions = len(self.action_space)
        self.title('Pac-Man')
        self.geometry('{0}x{1}+400+50'.format(Map_W * UNIT, Map_H * UNIT))

        # 2颗豆子 (row, col)
        self.bean_positions = list(BEAN_POSITIONS)
        self.bean_mask = 0
        # 3个静止幽灵 (row, col)
        self.ghosts = [
            {'row': 1, 'col': 2, 'type': 'static'},
            {'row': 2, 'col': 1, 'type': 'static'},
            {'row': 3, 'col': 3, 'type': 'static'},
        ]
        self._build_map()

    def _build_map(self):
        self.canvas = tk.Canvas(self, bg='white', height=Map_H * UNIT, width=Map_W * UNIT)
        for c in range(0, Map_W * UNIT, UNIT):
            self.canvas.create_line(c, 0, c, Map_H * UNIT)
        for r in range(0, Map_H * UNIT, UNIT):
            self.canvas.create_line(0, r, Map_W * UNIT, r)

        origin = np.array([UNIT / 2, UNIT / 2])
        IMG_SIZE = (80, 80)

        def load_img(path):
            return ImageTk.PhotoImage(Image.open(ASSET_DIR / path).resize(IMG_SIZE, Image.Resampling.LANCZOS))

        self.bm_beans = load_img("beans.png")
        self.bm_ghost = load_img("ghost.png")
        self.bm_person = load_img("pac-man.png")
        self.bm_flag = load_img("destination.png")

        # 终点 (4,4)
        self.flag = self.canvas.create_image(origin[0]+UNIT*4, origin[1]+UNIT*4,
                                              image=self.bm_flag, tag="destination")

        # 豆子
        self.bean_items = []
        for i, (row, col) in enumerate(self.bean_positions):
            cx, cy = grid_to_canvas(col, row)
            item = self.canvas.create_image(cx, cy, image=self.bm_beans, tag="bean%d" % i)
            self.bean_items.append(item)

        # 幽灵
        self.ghost_items = []
        for i, g in enumerate(self.ghosts):
            cx, cy = grid_to_canvas(g['col'], g['row'])
            item = self.canvas.create_image(cx, cy, image=self.bm_ghost, tag="ghost%d" % i)
            self.ghost_items.append(item)

        # 吃豆人，初始位置(0,0)
        cx, cy = grid_to_canvas(0, 0)
        self.person = self.canvas.create_image(cx, cy, image=self.bm_person)
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.1)
        self.canvas.delete(self.person)
        self.bean_mask = 0

        self.bean_positions = list(BEAN_POSITIONS)
        for item in self.bean_items:
            self.canvas.delete(item)
        self.bean_items = []
        for i, (row, col) in enumerate(self.bean_positions):
            cx, cy = grid_to_canvas(col, row)
            item = self.canvas.create_image(cx, cy, image=self.bm_beans, tag="bean%d" % i)
            self.bean_items.append(item)

        self.ghosts = [
            {'row': 1, 'col': 2, 'type': 'static'},
            {'row': 2, 'col': 1, 'type': 'static'},
            {'row': 3, 'col': 3, 'type': 'static'},
        ]
        for i, g in enumerate(self.ghosts):
            self.canvas.delete(self.ghost_items[i])
            cx, cy = grid_to_canvas(g['col'], g['row'])
            item = self.canvas.create_image(cx, cy, image=self.bm_ghost, tag="ghost%d" % i)
            self.ghost_items[i] = item

        cx, cy = grid_to_canvas(0, 0)
        self.person = self.canvas.create_image(cx, cy, image=self.bm_person)
        self.render()
        return self.get_state()

    def get_state(self):
        coords = self.canvas.coords(self.person)
        col = int(coords[0] / UNIT)
        row = int(coords[1] / UNIT)
        return int(map_state[row, col]), self.bean_mask

    def _get_pacman_grid_pos(self):
        coords = self.canvas.coords(self.person)
        col = min(int(coords[0] / UNIT), Map_W - 1)
        row = min(int(coords[1] / UNIT), Map_H - 1)
        return row, col

    def _check_ghost_collision(self):
        row, col = self._get_pacman_grid_pos()
        return (row, col) in GHOST_POSITIONS

    def step(self, action):
        """执行一个动作
        action: 0=上, 1=下, 2=左, 3=右
        """
        old_row, old_col = self._get_pacman_grid_pos()
        old_state = (grid_to_pos(old_row, old_col), self.bean_mask)
        next_state, reward, done = transition(old_state, action)
        new_row, new_col = pos_to_grid(next_state[0])

        self.canvas.move(self.person, (new_col - old_col) * UNIT, (new_row - old_row) * UNIT)

        old_mask = self.bean_mask
        self.bean_mask = next_state[1]
        for i, item in enumerate(self.bean_items):
            if not (old_mask & (1 << i)) and (self.bean_mask & (1 << i)):
                self.canvas.delete(item)

        return self.get_state(), reward, done

    def render(self):
        time.sleep(0.1)
        self.update()
        time.sleep(0.1)


def value_iteration(gamma=GAMMA, theta=1e-8, max_iterations=10000):
    """有模型强化学习方法：值迭代。"""
    values = {state: 0.0 for state in all_states()}
    policy = {}

    for _ in range(max_iterations):
        delta = 0.0
        for state in all_states():
            if is_terminal_state(state):
                continue

            old_v = values[state]
            action_values = []
            for action in range(len(ACTION_SPACE)):
                next_state, reward, done = transition(state, action)
                action_values.append(reward if done else reward + gamma * values[next_state])

            values[state] = max(action_values)
            policy[state] = int(np.argmax(action_values))
            delta = max(delta, abs(old_v - values[state]))

        if delta < theta:
            break

    return values, policy


def epsilon_greedy_action(q_values, state, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(len(ACTION_SPACE)))
    return int(np.argmax(q_values[state]))


def monte_carlo_control(episodes=5000, gamma=GAMMA, epsilon=1.0,
                        epsilon_min=0.05, seed=0, max_steps=200):
    """无模型强化学习方法：first-visit MC control。"""
    rng = np.random.default_rng(seed)
    q_values = defaultdict(lambda: np.zeros(len(ACTION_SPACE)))
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    metrics = {'returns': [], 'costs': [], 'path_lengths': [], 'success': []}

    for episode_idx in range(episodes):
        state = (START_POS, 0)
        episode = []
        eps = max(epsilon_min, epsilon * (1 - episode_idx / max(episodes, 1)))

        for _ in range(max_steps):
            action = epsilon_greedy_action(q_values, state, eps, rng)
            next_state, reward, done = transition(state, action)
            episode.append((state, action, reward))
            state = next_state
            if done:
                break

        total_return = sum(step[2] for step in episode)
        metrics['returns'].append(total_return)
        metrics['costs'].append(-total_return)
        metrics['path_lengths'].append(len(episode))
        metrics['success'].append(state[0] == GOAL_POS and state[1] == ALL_BEANS_MASK)

        g = 0.0
        returns = []
        for state, action, reward in reversed(episode):
            g = gamma * g + reward
            returns.append((state, action, g))

        visited = set()
        for state, action, g in reversed(returns):
            key = (state, action)
            if key in visited:
                continue
            visited.add(key)
            returns_sum[key] += g
            returns_count[key] += 1
            q_values[state][action] = returns_sum[key] / returns_count[key]

    policy = {state: int(np.argmax(q_values[state])) for state in all_states() if not is_terminal_state(state)}
    return q_values, policy, metrics


def run_policy(policy, max_steps=100):
    state = (START_POS, 0)
    path = [state]
    rewards = []

    for _ in range(max_steps):
        action = policy.get(state)
        if action is None:
            break
        state, reward, done = transition(state, action)
        path.append(state)
        rewards.append(reward)
        if done:
            break

    return path, rewards


def plot_training_curves(metrics, save_path="outputs/mc_curves.png"):
    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(metrics['costs'])
    axes[0].set_title('Total cost')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('-Return')
    axes[1].plot(metrics['path_lengths'])
    axes[1].set_title('Path length')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps')
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def print_path(path, rewards):
    rows = []
    for state in path:
        row, col = pos_to_grid(state[0])
        rows.append(f"({row},{col}) beans={state[1]:02b}")
    print(" -> ".join(rows))
    print("return:", sum(rewards), "steps:", len(rewards))


def print_policy(policy, bean_mask=0):
    print(f"policy for bean_mask={bean_mask:02b}")
    for row in range(Map_H):
        cells = []
        for col in range(Map_W):
            state = (grid_to_pos(row, col), bean_mask)
            if (row, col) in GHOST_POSITIONS:
                cells.append("G")
            elif state[0] == GOAL_POS:
                cells.append("T")
            else:
                cells.append(ACTION_ARROWS[policy[state]])
        print(" ".join(cells))


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_dp_command(config):
    _, policy = value_iteration(gamma=config["dp"]["gamma"])
    print_policy(policy, bean_mask=0)
    path, rewards = run_policy(policy)
    print_path(path, rewards)


def run_mc_command(config):
    mc_config = config["mc"]
    _, policy, metrics = monte_carlo_control(
        episodes=mc_config["episodes"],
        gamma=mc_config["gamma"],
        epsilon=mc_config["epsilon"],
        epsilon_min=mc_config["epsilon_min"],
        seed=mc_config["seed"],
        max_steps=mc_config["max_steps"],
    )
    curve_path = plot_training_curves(metrics, mc_config["output"])
    path, rewards = run_policy(policy)
    print("saved:", curve_path)
    print_path(path, rewards)


def parse_args():
    parser = argparse.ArgumentParser(description="MiniPac-RL")
    parser.add_argument("--config", default=CONFIG_PATH, help="path to YAML config file")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("dp", help="run value iteration")
    subparsers.add_parser("mc", help="run Monte Carlo control")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.command == "dp":
        run_dp_command(load_config(args.config))
    elif args.command == "mc":
        run_mc_command(load_config(args.config))
    else:
        env = Map()
        env.mainloop()
