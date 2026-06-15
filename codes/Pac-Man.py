import numpy as np
import time
import tkinter as tk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt


UNIT = 100
Map_H = 5
Map_W = 5

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


class Map(tk.Tk, object):
    def __init__(self):
        super(Map, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('Pac-Man')
        self.geometry('{0}x{1}+400+50'.format(Map_W * UNIT, Map_H * UNIT))

        # 2颗豆子 (row, col)
        self.bean_positions = [(0, 2), (2, 3)]
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
            return ImageTk.PhotoImage(Image.open(path).resize(IMG_SIZE, Image.Resampling.LANCZOS))

        self.bm_beans = load_img("./beans.png")
        self.bm_ghost = load_img("./ghost.png")
        self.bm_person = load_img("./pac-man.png")
        self.bm_flag = load_img("./destination.png")

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
        origin = np.array([UNIT / 2, UNIT / 2])

        self.bean_positions = [(0, 2), (2, 3)]
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
        return map_state[row, col]

    def _get_pacman_grid_pos(self):
        coords = self.canvas.coords(self.person)
        col = min(int(coords[0] / UNIT), Map_W - 1)
        row = min(int(coords[1] / UNIT), Map_H - 1)
        return row, col

    def _check_ghost_collision(self):
        row, col = self._get_pacman_grid_pos()
        for g in self.ghosts:
            if row == g['row'] and col == g['col']:
                return True
        return False

    def step(self, action):
        """执行一个动作
        action: 0=上, 1=下, 2=左, 3=右
        """
        s = self.canvas.coords(self.person)
        base_action = np.array([0, 0])
        cost = -1  # 每走一步基础代价

        if action == 0:  # 上
            if s[1] >= UNIT:
                base_action[1] -= UNIT
            else:
                cost = -10  # 碰壁惩罚
        elif action == 1:  # 下
            if s[1] < (Map_H - 1) * UNIT:
                base_action[1] += UNIT
            else:
                cost = -10
        elif action == 2:  # 左
            if s[0] >= UNIT:
                base_action[0] -= UNIT
            else:
                cost = -10
        elif action == 3:  # 右
            if s[0] < (Map_W - 1) * UNIT:
                base_action[0] += UNIT
            else:
                cost = -10

        self.canvas.move(self.person, base_action[0], base_action[1])
        row, col = self._get_pacman_grid_pos()

        # TODO 1: 吃到豆子，给予相应奖励，并从画布上移除该豆子

        if self._check_ghost_collision():
            # TODO 2: 碰撞幽灵，给予惩罚，并结束回合
            pass

        # TODO 3: 到达终点，结束回合。（到达终点时豆子有没有吃完？）

        return self.get_state(), cost, False

    def render(self):
        time.sleep(0.1)
        self.update()
        time.sleep(0.1)

# TODO 4：有模型强化学习方法（值迭代），给出最优策略
# TODO 5：无模型强化学习方法（蒙特卡洛），给出最优策略及学习过程


if __name__ == '__main__':
    env = Map()
    env.mainloop()
