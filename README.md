# MiniPac-RL

本项目是 5x5 简化吃豆人强化学习实验。代码尽量沿用助教给出的 `codes/Pac-Man.py`，在同一文件中补全环境交互、动态规划 value iteration 和蒙特卡洛控制。

## 环境准备

```bash
uv sync
```

## 1. 检查 GUI 地图

```bash
python codes/Pac-Man.py
```

应看到 5x5 地图、两颗豆子、三个静止幽灵、右下角终点和左上角吃豆人。

## 2. 运行动态规划

```bash
python codes/Pac-Man.py dp
```

该命令会：

- 使用已知环境模型运行 value iteration。
- 打印初始 `bean_mask=00` 下每个格子的最优动作。
- 打印从起点按照最优策略执行得到的路径、总回报和步数。

可调整折扣因子：

编辑 `codes/config.yaml` 中的 `dp.gamma`，然后重新运行 `python codes/Pac-Man.py dp`。

## 3. 运行蒙特卡洛训练

先做一个小规模 smoke test：先把 `codes/config.yaml` 中的 `mc.episodes` 临时改为 `100`，再运行：

```bash
python codes/Pac-Man.py mc
```

确认能正常结束并生成曲线后，再改回较大的 episode 数运行正式实验：

编辑 `codes/config.yaml` 中的 `mc.episodes`、`mc.seed` 和 `mc.output`，然后重新运行 `python codes/Pac-Man.py mc`。

该命令会：

- 使用 first-visit Monte Carlo control 学习动作价值。
- 使用 epsilon-greedy 探索，默认 `epsilon` 从 `1.0` 衰减到 `0.05`。
- 保存总代价 `-Return` 和路径长度曲线。
- 打印训练后贪心策略从起点执行得到的路径。

常用可调参数：

```yaml
mc:
  episodes: 10000
  epsilon: 1.0
  epsilon_min: 0.05
  gamma: 0.95
  seed: 1
  max_steps: 200
  output: outputs/mc_curves_seed1.png
```

如果想临时使用另一份配置文件：

```bash
python codes/Pac-Man.py --config codes/config.yaml mc
```

## 4. 建议实验流程

1. 运行 `python -m compileall codes`，确认代码语法正确。
2. 运行 `python codes/Pac-Man.py dp`，记录最优策略和最优路径。
3. 将 `codes/config.yaml` 中的 `mc.episodes` 暂时改为 `100`，运行 `python codes/Pac-Man.py mc`，确认训练流程和绘图正常。
4. 分别使用 `5000`、`10000` 等 episode 数运行蒙特卡洛实验，观察总代价和路径长度是否整体下降。
5. 可改变 `mc.seed`、`mc.epsilon_min` 或 `mc.gamma` 做对比实验。

报告中可使用的结果包括：DP 最优策略、DP 路径、MC 曲线图、不同 episode/seed 下的对比曲线，以及最终学到的策略路径。

## 状态与奖励

状态为 `(position, bean_mask)`：

- `position` 是 0 到 24 的格子编号。
- `bean_mask` 记录两颗豆子是否已经吃到，例如 `00` 表示都没吃，`11` 表示都吃完。

奖励设置：

- 普通移动：`-1`
- 撞墙：`-10`
- 第一次吃到豆子：额外 `+10`
- 撞幽灵：`-100`
- 未吃完豆子到达终点：`-100`
- 吃完豆子到达终点：`+100`

