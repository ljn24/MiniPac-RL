import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import tkinter as tk

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import yaml


UNIT = 100
Map_H = 5
Map_W = 5

START_POS = 0
GOAL_POS = 24

# 状态为 (position, bean_mask)，bean_mask 记录已经吃到的豆子。
BEAN_POSITIONS = [(0, 2), (2, 3)]
GHOST_POSITIONS = [(1, 2), (2, 1), (3, 3)]
ACTION_SPACE = ['u', 'd', 'l', 'r']
ACTION_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_ARROWS = ['^', 'v', '<', '>']

STEP_REWARD = -1
WALL_REWARD = -10
BEAN_REWARD = 10
GHOST_REWARD = -100
FAIL_REWARD = -100
SUCCESS_REWARD = 100

# 较大的终止奖励用于鼓励先吃完豆子再到达终点。
GAMMA = 0.95
ALL_BEANS_MASK = (1 << len(BEAN_POSITIONS)) - 1
ASSET_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ASSET_DIR / "config.yaml"


def grid_to_canvas(col, row):
    return UNIT / 2 + col * UNIT, UNIT / 2 + row * UNIT


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
    """DP 使用的确定性环境模型，MC 也通过它采样交互。"""
    pos, bean_mask = state
    row, col = pos_to_grid(pos)
    dr, dc = ACTION_DELTAS[action]
    nr, nc = row + dr, col + dc

    # 撞墙时位置不变，但仍给予惩罚。
    if not (0 <= nr < Map_H and 0 <= nc < Map_W):
        return (pos, bean_mask), WALL_REWARD, False

    next_pos = grid_to_pos(nr, nc)
    reward = STEP_REWARD
    next_mask = bean_mask

    for i, bean in enumerate(BEAN_POSITIONS):
        if (nr, nc) == bean and not (bean_mask & (1 << i)):
            # 每颗豆子只有第一次吃到时才给予奖励。
            next_mask |= 1 << i
            reward += BEAN_REWARD

    if (nr, nc) in GHOST_POSITIONS:
        return (next_pos, next_mask), GHOST_REWARD, True

    if next_pos == GOAL_POS:
        # 只有吃完全部豆子后到达终点才算成功。
        reward = SUCCESS_REWARD if next_mask == ALL_BEANS_MASK else FAIL_REWARD
        return (next_pos, next_mask), reward, True

    return (next_pos, next_mask), reward, False


class Map(tk.Tk):
    def __init__(self):
        super().__init__()
        self.action_space = ACTION_SPACE
        self.n_actions = len(self.action_space)
        self.title('Pac-Man')
        self.geometry(f'{Map_W * UNIT}x{Map_H * UNIT}+400+50')

        # 2颗豆子 (行, 列)
        self.bean_positions = list(BEAN_POSITIONS)
        self.bean_mask = 0
        self._build_map()

    def _build_map(self):
        self.canvas = tk.Canvas(self, bg='white', height=Map_H * UNIT, width=Map_W * UNIT)
        for c in range(0, Map_W * UNIT, UNIT):
            self.canvas.create_line(c, 0, c, Map_H * UNIT)
        for r in range(0, Map_H * UNIT, UNIT):
            self.canvas.create_line(0, r, Map_W * UNIT, r)

        IMG_SIZE = (80, 80)

        def load_img(path):
            return ImageTk.PhotoImage(Image.open(ASSET_DIR / path).resize(IMG_SIZE, Image.Resampling.LANCZOS))

        self.bm_beans = load_img("beans.png")
        self.bm_ghost = load_img("ghost.png")
        self.bm_person = load_img("pac-man.png")
        self.bm_flag = load_img("destination.png")

        # 终点 (4,4)
        self.flag = self.canvas.create_image(*grid_to_canvas(4, 4),
                                              image=self.bm_flag, tag="destination")

        self.bean_items = self._create_items(self.bean_positions, self.bm_beans, "bean")
        self.ghost_items = self._create_items(GHOST_POSITIONS, self.bm_ghost, "ghost")

        # 吃豆人，初始位置(0,0)
        cx, cy = grid_to_canvas(0, 0)
        self.person = self.canvas.create_image(cx, cy, image=self.bm_person)
        self.canvas.pack()

    def _create_items(self, positions, image, tag_prefix):
        items = []
        for i, (row, col) in enumerate(positions):
            items.append(self.canvas.create_image(
                *grid_to_canvas(col, row),
                image=image,
                tag=f"{tag_prefix}{i}",
            ))
        return items

    def reset(self):
        self.update()
        time.sleep(0.1)
        self.canvas.delete(self.person)
        self.bean_mask = 0

        # 被吃掉的豆子图片会被删除，因此重置时需要重新创建。
        self.bean_positions = list(BEAN_POSITIONS)
        for item in self.bean_items:
            self.canvas.delete(item)
        self.bean_items = self._create_items(self.bean_positions, self.bm_beans, "bean")

        for item in self.ghost_items:
            self.canvas.delete(item)
        self.ghost_items = self._create_items(GHOST_POSITIONS, self.bm_ghost, "ghost")

        cx, cy = grid_to_canvas(0, 0)
        self.person = self.canvas.create_image(cx, cy, image=self.bm_person)
        self.render()
        return self.get_state()

    def get_state(self):
        coords = self.canvas.coords(self.person)
        col = int(coords[0] / UNIT)
        row = int(coords[1] / UNIT)
        return grid_to_pos(row, col), self.bean_mask

    def _get_pacman_grid_pos(self):
        coords = self.canvas.coords(self.person)
        col = min(int(coords[0] / UNIT), Map_W - 1)
        row = min(int(coords[1] / UNIT), Map_H - 1)
        return row, col

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

    # 对所有非终止状态执行 Bellman 最优备份。
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

        # 当价值函数基本收敛时停止迭代。
        if delta < theta:
            break

    return values, policy


def epsilon_greedy_action(q_values, state, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(len(ACTION_SPACE)))
    return int(np.argmax(q_values[state]))


def monte_carlo_control(episodes=5000, gamma=GAMMA, epsilon=1.0,
                        epsilon_min=0.05, seed=0, max_steps=200,
                        record_episodes=None):
    """无模型强化学习方法：首次访问蒙特卡洛控制。"""
    rng = np.random.default_rng(seed)
    record_episodes = set(record_episodes or [])
    q_values = defaultdict(lambda: np.zeros(len(ACTION_SPACE)))
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    metrics = {
        'returns': [], 'costs': [], 'path_lengths': [],
        'success': [], 'epsilons': [], 'recorded_episodes': []
    }

    for episode_idx in range(episodes):
        state = (START_POS, 0)
        episode = []
        # 线性衰减 epsilon：前期充分探索，后期策略更稳定。
        eps = max(epsilon_min, epsilon * (1 - episode_idx / max(episodes, 1)))

        for _ in range(max_steps):
            action = epsilon_greedy_action(q_values, state, eps, rng)
            next_state, reward, done = transition(state, action)
            episode.append((state, action, reward, next_state))
            state = next_state
            if done:
                break

        total_return = sum(step[2] for step in episode)
        metrics['returns'].append(total_return)
        metrics['costs'].append(-total_return)
        metrics['path_lengths'].append(len(episode))
        metrics['success'].append(state[0] == GOAL_POS and state[1] == ALL_BEANS_MASK)
        metrics['epsilons'].append(eps)

        # 保存指定训练轮次的真实轨迹，用于生成学习过程视频。
        if episode_idx + 1 in record_episodes:
            metrics['recorded_episodes'].append({
                'episode': episode_idx + 1,
                'path': [(START_POS, 0)] + [step[3] for step in episode],
                'rewards': [step[2] for step in episode],
            })

        g = 0.0
        returns = []
        # 反向计算折扣回报，用于首次访问 MC 更新。
        for state, action, reward, _ in reversed(episode):
            g = gamma * g + reward
            returns.append((state, action, g))

        visited = set()
        for state, action, g in reversed(returns):
            key = (state, action)
            if key in visited:
                continue
            visited.add(key)
            # Q(s,a) 取该状态动作对历史回报的平均值。
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
    # 报告中使用这两条曲线展示学习过程。
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


def render_state_image(state, assets, cell_size=100):
    image = Image.new("RGB", (Map_W * cell_size, Map_H * cell_size), "white")
    draw = ImageDraw.Draw(image)
    # 离屏渲染视频帧，因此不需要额外录屏软件。
    for x in range(0, (Map_W + 1) * cell_size, cell_size):
        draw.line((x, 0, x, Map_H * cell_size), fill="black", width=2)
    for y in range(0, (Map_H + 1) * cell_size, cell_size):
        draw.line((0, y, Map_W * cell_size, y), fill="black", width=2)

    pos, bean_mask = state

    def paste_asset(name, row, col):
        asset = assets[name]
        x = col * cell_size + (cell_size - asset.width) // 2
        y = row * cell_size + (cell_size - asset.height) // 2
        image.paste(asset, (x, y), asset)

    paste_asset("destination", 4, 4)
    for i, (row, col) in enumerate(BEAN_POSITIONS):
        if not (bean_mask & (1 << i)):
            paste_asset("beans", row, col)
    for row, col in GHOST_POSITIONS:
        paste_asset("ghost", row, col)

    row, col = pos_to_grid(pos)
    paste_asset("pacman", row, col)
    return image


def load_video_assets(size=80):
    names = {
        "beans": "beans.png",
        "ghost": "ghost.png",
        "pacman": "pac-man.png",
        "destination": "destination.png",
    }
    return {
        name: Image.open(ASSET_DIR / filename).convert("RGBA").resize(
            (size, size), Image.Resampling.LANCZOS
        )
        for name, filename in names.items()
    }


def write_episode_frames(frames, path, assets, hold_frames=3, pause_frames=6, max_steps=40):
    trimmed = path[:max_steps + 1]
    for state in trimmed:
        frame = render_state_image(state, assets)
        frames.extend([frame] * hold_frames)
    # 在终止状态短暂停留，便于观察失败或成功位置。
    frames.extend([render_state_image(trimmed[-1], assets)] * pause_frames)


def encode_video(frames, save_path, fps=12):
    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for i, frame in enumerate(frames):
            frame.save(tmpdir / f"frame_{i:05d}.png")
        # ffmpeg 只用于把 PNG 帧序列编码为 MP4。
        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", str(tmpdir / "frame_%05d.png"),
            "-pix_fmt", "yuv420p", str(output),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output


def format_path(path):
    rows = []
    for state in path:
        row, col = pos_to_grid(state[0])
        rows.append(f"({row},{col}) beans={state[1]:02b}")
    return " -> ".join(rows)


def policy_grid_lines(policy, bean_mask=0):
    lines = [f"policy for bean_mask={bean_mask:02b}"]
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
        lines.append(" ".join(cells))
    return lines


def save_dp_result(policy, path, rewards, gamma, save_path):
    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write("Dynamic programming result\n")
        f.write(f"gamma: {gamma}\n")
        f.write("\n".join(policy_grid_lines(policy, bean_mask=0)))
        f.write("\n\npath:\n")
        f.write(format_path(path))
        f.write("\n\nrewards:\n")
        f.write(", ".join(str(reward) for reward in rewards))
        f.write(f"\nreturn: {sum(rewards)}\n")
        f.write(f"steps: {len(rewards)}\n")
        f.write(f"success: {path[-1][0] == GOAL_POS and path[-1][1] == ALL_BEANS_MASK}\n")
    return output


def save_mc_metrics(metrics, save_path):
    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "epsilon", "return", "cost", "path_length", "success"])
        for episode_idx in range(len(metrics["returns"])):
            writer.writerow([
                episode_idx + 1,
                metrics["epsilons"][episode_idx],
                metrics["returns"][episode_idx],
                metrics["costs"][episode_idx],
                metrics["path_lengths"][episode_idx],
                int(metrics["success"][episode_idx]),
            ])
    return output


def save_mc_summary(metrics, path, rewards, mc_config, save_path, window=500):
    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    # 最近窗口统计比原始 CSV 更适合在报告中引用。
    start = max(0, len(metrics["returns"]) - window)
    window_size = len(metrics["returns"]) - start
    recent_success_rate = (
        sum(metrics["success"][start:]) / window_size if window_size else 0.0
    )
    recent_avg_cost = (
        sum(metrics["costs"][start:]) / window_size if window_size else 0.0
    )
    recent_avg_path_length = (
        sum(metrics["path_lengths"][start:]) / window_size if window_size else 0.0
    )
    summary = {
        "config": mc_config,
        "episodes": len(metrics["returns"]),
        "recent_window": window_size,
        "recent_success_rate": recent_success_rate,
        "recent_avg_cost": recent_avg_cost,
        "recent_avg_path_length": recent_avg_path_length,
        "greedy_path": format_path(path),
        "greedy_rewards": rewards,
        "greedy_return": sum(rewards),
        "greedy_steps": len(rewards),
        "greedy_success": path[-1][0] == GOAL_POS and path[-1][1] == ALL_BEANS_MASK,
    }
    with open(output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return output


def print_path(path, rewards):
    print(format_path(path))
    print("return:", sum(rewards), "steps:", len(rewards))


def print_policy(policy, bean_mask=0):
    print("\n".join(policy_grid_lines(policy, bean_mask)))


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_dp_command(config):
    dp_config = config["dp"]
    _, policy = value_iteration(gamma=dp_config["gamma"])
    print_policy(policy, bean_mask=0)
    path, rewards = run_policy(policy)
    print_path(path, rewards)
    if dp_config.get("output"):
        result_path = save_dp_result(policy, path, rewards, dp_config["gamma"], dp_config["output"])
        print("saved:", result_path)


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
    if mc_config.get("metrics_output"):
        metrics_path = save_mc_metrics(metrics, mc_config["metrics_output"])
        print("saved:", metrics_path)
    if mc_config.get("summary_output"):
        summary_path = save_mc_summary(metrics, path, rewards, mc_config, mc_config["summary_output"])
        print("saved:", summary_path)
    print_path(path, rewards)


def run_video_command(config):
    mc_config = config["mc"]
    video_config = config.get("video", {})
    episodes = mc_config["episodes"]

    # 渲染固定训练轮次，最后追加最终贪心策略。
    checkpoints = video_config.get("checkpoints") or [1, 100, 500, 1000, episodes]
    checkpoints = sorted({max(1, min(episodes, int(ep))) for ep in checkpoints})

    _, policy, metrics = monte_carlo_control(
        episodes=episodes,
        gamma=mc_config["gamma"],
        epsilon=mc_config["epsilon"],
        epsilon_min=mc_config["epsilon_min"],
        seed=mc_config["seed"],
        max_steps=mc_config["max_steps"],
        record_episodes=checkpoints,
    )
    greedy_path, _ = run_policy(policy)

    assets = load_video_assets(video_config.get("asset_size", 80))
    frames = []
    selected_paths = [item["path"] for item in metrics["recorded_episodes"]]
    selected_paths.append(greedy_path)
    for path in selected_paths:
        write_episode_frames(
            frames,
            path,
            assets,
            hold_frames=video_config.get("hold_frames", 3),
            pause_frames=video_config.get("pause_frames", 6),
            max_steps=video_config.get("max_steps_per_episode", 40),
        )
    if not frames:
        raise RuntimeError("no frames generated for video")

    duration = len(frames) / video_config.get("fps", 12)
    print(f"checkpoints: {checkpoints}")
    print(f"segments: {len(selected_paths)} duration: {duration:.1f}s")

    video_path = encode_video(
        frames,
        video_config.get("output", "outputs/learning_process.mp4"),
        fps=video_config.get("fps", 12),
    )
    print("saved:", video_path)


def parse_args():
    parser = argparse.ArgumentParser(description="MiniPac-RL")
    parser.add_argument("--config", default=CONFIG_PATH, help="path to YAML config file")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("dp", help="run value iteration")
    subparsers.add_parser("mc", help="run Monte Carlo control")
    subparsers.add_parser("video", help="render a Monte Carlo learning process video")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.command == "dp":
        run_dp_command(load_config(args.config))
    elif args.command == "mc":
        run_mc_command(load_config(args.config))
    elif args.command == "video":
        run_video_command(load_config(args.config))
    else:
        env = Map()
        env.mainloop()
