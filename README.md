# MiniPac-RL

5x5 简化吃豆人强化学习实验。吃豆人从左上角出发，需要吃掉两颗豆子，避开三个静止幽灵，并到达右下角终点。

实现内容：

- 🧭 环境建模：状态、动作、转移和奖励。
- 🧮 动态规划：value iteration。
- 🎲 蒙特卡洛控制：first-visit Monte Carlo control + epsilon-greedy。
- 🎬 学习过程视频：按指定训练轮次渲染轨迹动画。

## 🚀 运行环境

```bash
uv sync
source .venv/bin/activate
```

项目依赖写在 `pyproject.toml` 中，`uv sync` 会自动创建 `.venv` 并安装依赖。

## ▶️ 使用方式

查看 GUI 初始地图：

```bash
python codes/Pac-Man.py
```

运行动态规划：

```bash
python codes/Pac-Man.py dp
```

运行蒙特卡洛训练：

```bash
python codes/Pac-Man.py mc
```

生成学习过程视频：

```bash
python codes/Pac-Man.py video
```

## 📁 输出文件

运行后会在 `outputs/` 下生成实验材料：

- `dp_result.txt`：DP 最优策略、路径、奖励序列和回报。
- `mc_curves.png`：MC 训练总代价和路径长度曲线。
- `mc_metrics.csv`：每轮训练的 return、cost、path length、success 等数据。
- `mc_summary.json`：最终贪心策略和最近窗口统计。
- `learning_process.mp4`：学习过程可视化视频。

## ⚙️ 配置

主要参数位于 `codes/config.yaml`：

- `dp.gamma`：动态规划折扣因子。
- `mc.episodes`：MC 训练轮数。
- `mc.seed`：随机种子。
- `mc.epsilon` / `mc.epsilon_min`：探索率及其下限。
- `video.checkpoints`：视频中展示的训练轮次。
- `video.fps` / `video.hold_frames`：视频播放速度。

如需使用其它配置文件：

```bash
python codes/Pac-Man.py --config codes/config.yaml mc
```

## 🧩 状态与奖励

状态为 `(position, bean_mask)`：

- `position`：0 到 24 的格子编号。
- `bean_mask`：两颗豆子的收集情况，例如 `00` 表示都未吃，`11` 表示都已吃。

奖励：

- 普通移动：`-1`
- 撞墙：`-10`
- 第一次吃到豆子：额外 `+10`
- 撞幽灵：`-100`
- 未吃完豆子到达终点：`-100`
- 吃完豆子到达终点：`+100`
