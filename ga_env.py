import sys
import numpy as np
import time
import math
import copy
from gym import spaces
from gym.utils import seeding
import random
import math
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
import json

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False
matplotlib.use('TkAgg')  # 或其他后端

#   以v in [0,1],theta in [-1,1]为动作
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk
'''
地图大小为400*400
无人机数量N_UAV=1
服务点数量为100，所有服务点在限定区域内随机分布，服务点具有不同数据收集特性
固定地图
'''
WIDTH = 10  # 地图的宽度
HEIGHT = 10  # 地图的高度
UNIT = 100  # 每个方块的大小（像素值）   400m*400m的地图
# C = 5000  # 传感器的容量都假设为5000
P_u = pow(10, -5)  # 传感器的发射功率 0.01mW,-20dbm
P_d = 15  # 无人机下行发射功率 10W,40dBm
H = 20.  # 无人机固定悬停高度 10m
B = pow(10, 6)
R_d = 20.  # 无人机充电覆盖范围 10m能接受到0.1mW,30m能接收到0.01mW
N_S_ = 100  # 设备个数
V = 35  # 无人机最大速度 20m/s
V_max = 35.0
V_me = 10.0
EC_min = 126.
EC_grd = 506.69  # 20:178. 30:356.26  35:506.69

# 非线性能量接收机模型
Mj = 9.079 * pow(10, -6)
aj = 47083
bj = 2.9 * pow(10, -6)
Oj = 1 / (1 + math.exp(aj * bj))

np.random.seed(1)

# 加载设备访问序列
def load_paths(filename="best_paths6.json"):
    with open(filename, 'r') as f:
        return json.load(f)

# 定义无人机类
class UAV(tk.Tk, object):
    def __init__(self, R_dc=20.):
        super(UAV, self).__init__()
        # POI位置
        self.N_POI = N_S_  # 传感器数量
        self.dis = np.zeros(self.N_POI)  # 距离的平方
        self.elevation = np.zeros(self.N_POI)  # 仰角
        self.pro = np.zeros(self.N_POI)  # 视距概率
        self.h = np.zeros(self.N_POI)  # 信道增益
        self.N_UAV = 1
        self.max_speed = V  # 无人机最大速度 20m/s
        self.H = 20.  # 无人机飞行高度 10m
        self.X_min = 0
        self.Y_min = 0
        self.round = -1  # # 选择第几回合 主要用于评估
        self.X_max = (WIDTH) * UNIT
        self.Y_max = (HEIGHT) * UNIT  # 地图边界
        self.R_dc = R_dc  # 水平覆盖距离 10m
        self.sdc = math.sqrt(pow(self.R_dc, 2) + pow(self.H, 2))  # 最大DC服务距离
        self.noise = pow(10, -12)  # 噪声功率为-90dbm
        self.E = 100000
        self.AutoUAV = []
        self.Aim = []
        self.Pass = []
        self.N_AIM = 1  # 选择服务的用户数
        self.FX = 0.

        self.data = load_paths()
        self.paths = self.data["best_paths"]  # 所有回合路径（列表的列表）
        self.now_id = 1  # 用于选择目标设备
        self.SoPcenter = np.array(self.data["SoPcenter"])  # 提取设备位置
        self.action_space = spaces.Box(low=np.array([0., -1.]), high=np.array([1., 1.]),
                                       dtype=np.float32)  # 以 v in [0,1],theta in [-1,1]为动作
        self.state_dim = 6  # 状态空间为 目标用户位置与无人机的相对位置，无人机位置，是否撞墙
        self.state = np.zeros(self.state_dim)
        self.xy = np.array([[0, 0]])
        self.xy0 = np.array([[0, 0]])
        self.b_S = np.array(self.data["b_S"])  # 提取数据量

        self.remaining_time = 600
        self.coord_to_id = {tuple(coord): i for i, coord in
                            enumerate(np.vstack([self.SoPcenter, self.xy]))}  # 创建坐标到ID的映射字典

        self.title('MAP')
        self.geometry('{0}x{1}'.format(WIDTH * UNIT, HEIGHT * UNIT))  # Tkinter 的几何形状
        self.build_maze()

    # 创建地图
    def build_maze(self):
        # 创建画布 Canvas.白色背景，宽高。
        self.canvas = tk.Canvas(self, bg='white', width=WIDTH * UNIT, height=HEIGHT * UNIT)

        # 创建用户
        for i in range(self.N_POI):
            # 创建椭圆，指定起始位置。填充颜色
            self.canvas.create_oval(
                self.SoPcenter[i][0] - 2, self.SoPcenter[i][1] - 2,
                self.SoPcenter[i][0] + 2, self.SoPcenter[i][1] + 2,
                fill='pink')

        # 创建无人机
        self.xy = np.random.randint(0, 1, size=[self.N_UAV, 2])
        for i in range(self.N_UAV):
            L_UAV = self.canvas.create_oval(
                self.xy[i][0] - R_d, self.xy[i][1] - R_d,
                self.xy[i][0] + R_d, self.xy[i][1] + R_d,
                fill='yellow')
            self.AutoUAV.append(L_UAV)

        # # 初始化状态空间值
        # idx_target = self.get_cache(self.target[self.now_id])
        # pxy = self.SoPcenter[idx_target]
        # L_AIM = self.canvas.create_rectangle(
        #     pxy[0] - 5, pxy[1] - 5,
        #     pxy[0] + 5, pxy[1] + 5,
        #     fill='red')
        # self.Aim.append(L_AIM)

        self.canvas.pack()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # 重置，随机初始化无人机的位置
    def reset(self):
        self.render()

        for i in range(self.N_UAV):
            self.canvas.delete(self.AutoUAV[i])
        self.AutoUAV = []
        self.Pass = []
        # now = [0, 1, 8, 13, 19, 24, 25, 27, 29, 30, 31, 38, 43, 49, 54, 55, 57, 59, 60, 61, 68, 73, 79, 84, 85, 87, 89, 90, 91, 98]
        now = [0, 1, 2, 3, 4, 5, 6, 11, 15, 16, 17, 32, 42, 44, 46, 48, 50, 56, 58, 66, 67, 72, 76, 79, 81, 83, 84, 85, 95, 99]
        # now = [0, 1, 3, 4, 11, 14, 15, 18, 27, 42, 44, 45, 46, 48, 49, 50, 56, 58, 61, 66, 67, 71, 76, 78, 79, 81, 85, 93, 95, 99]
        # print(len(now))
        self.round += 1 # # 选择第几回合 主要用于评估33,18,
        self.target = self.paths[now[self.round % 30]]  # 第一轮的最优设备访问序列
        # print(self.round % 53)
        # self.target = self.paths[self.round % 100]  # 第一轮的最优设备访问序列


        for i in range(len(self.Aim)):
            self.canvas.delete(self.Aim[i])
        self.Aim = []
        self.remaining_time = 600
        self.E = 100000
        self.t_need = 0  # 现位置到原点需要的时间
        self.E_need = 0

        # 随机初始化无人机位置
        self.xy = np.random.randint(0, 1, size=[self.N_UAV, 2]).astype(float)
        for i in range(self.N_UAV):
            L_UAV = self.canvas.create_oval(
                self.xy[i][0] - R_d, self.xy[i][1] - R_d,
                self.xy[i][0] + R_d, self.xy[i][1] + R_d,
                fill='yellow')
            self.AutoUAV.append(L_UAV)
        self.FX = 0
        self.now_id = 1
        self.b_S = np.array(self.data["b_S"]) * 1000000  # 提取数据量
        self.Ec = 0
        # 初始化状态空间值
        self.idx_target = self.target[self.now_id]
        self.pxy = self.SoPcenter[self.idx_target]
        L_AIM = self.canvas.create_rectangle(
            self.pxy[0] - 5, self.pxy[1] - 5,
            self.pxy[0] + 5, self.pxy[1] + 5,
            fill='red')
        self.Aim.append(L_AIM)

        self.state = np.concatenate(((self.pxy - self.xy[0]).flatten() / 1000., self.xy.flatten() / 1000., [0.,self.Ec /1000]))

        return self.state

    # 传入当前状态和输入动作输出下一个状态和奖励
    def step_move(self, action, above=False, count="0"):
        gohome, v2 = 0, 0
        e2 = EC_min
        EcH, EcE = 0, 0
        if above == True:
            detX = action[:self.N_UAV] * self.max_speed
            detY = action[self.N_UAV:] * self.max_speed
        else:
            detX = action[0] * self.max_speed * math.cos(action[1] * math.pi) * 3
            detY = action[0] * self.max_speed * math.sin(action[1] * math.pi) * 3
        state_ = np.zeros(self.state_dim)
        xy_ = copy.deepcopy(self.xy)  # 位置更新
        Flag = False  # 无人机是否飞行标识
        for i in range(self.N_UAV):  # 无人机位置更新
            xy_[i][0] += detX
            xy_[i][1] += detY
            # 当无人机更新后的位置超出地图范围时
            if xy_[i][0] >= self.X_min and xy_[i][0] <= self.X_max:
                if xy_[i][1] >= self.Y_min and xy_[i][1] <= self.Y_max:
                    self.FX = 0.
                    Flag = True
                else:
                    xy_[i][0] -= detX
                    xy_[i][1] -= detY
                    self.FX = 1.
            else:
                xy_[i][0] -= detX
                xy_[i][1] -= detY
                self.FX = 1.
        dis = np.linalg.norm(xy_ - self.xy0)
        if Flag:
            if count == "1":
                v2 = V_max
                e2 = EC_grd
                ft = math.sqrt((action[0] * self.max_speed) ** 2 + (action[1] * self.max_speed) ** 2) / V_max
                # ft = ft * 3 # 飞行时间
                ec = EC_grd * ft  # 飞行能耗
                self.t_need = dis / V_max  # 现位置到原点需要的时间
                self.E_need = EC_grd * self.t_need
            elif count == "2":
                v2 = V_me
                ft = math.sqrt((action[0] * self.max_speed) ** 2 + (action[1] * self.max_speed) ** 2) / V_me  # 飞行时间
                # ft = ft * 3
                ec = EC_min * ft  # 飞行能耗
                self.t_need = dis / V_me  # 现位置到原点需要的时间
                self.E_need = EC_min * self.t_need
            else:
                ft = 3
                V = math.sqrt(pow(detX, 2) + pow(detY, 2)) / 3
                v2 = V
                ec = 79.86 * (1 + 0.000208 * pow(V, 2)) + 88.63 * math.sqrt(
                    math.sqrt(1 + pow(V, 4) / 1055.0673312400002) - pow(V, 2) / 32.4818) + 0.009242625 * pow(V, 3)
                ec = ec * 3
                self.t_need = dis / V_me  # 现位置到原点需要的时间
                self.E_need = EC_min * self.t_need
        else:
            ec = 168.49  # 悬停能耗
            ft = 3
            ec = ec * 3
            self.t_need = dis / V_me  # 现位置到原点需要的时间
            self.E_need = EC_min * self.t_need
        EcF = ec
        self.E -= ec
        self.Ec = EcF
        self.remaining_time -= ft

        if self.E < self.E_need:
            self.E += ec
            xy_ = self.xy
            dis = np.linalg.norm(xy_ - self.xy0)
            self.t_need = dis / v2  # 现位置到原点需要的时间
            self.E_need = e2 * self.t_need
            EcF = 0
            gohome = 1

        if self.remaining_time < self.t_need:
            self.remaining_time += self.t_need
            xy_ = self.xy
            dis = np.linalg.norm(xy_ - self.xy0)
            self.t_need = dis / v2  # 现位置到原点需要的时间
            self.E_need = e2 * self.t_need
            EcF = 0
            gohome = 1

        for i in range(self.N_UAV):
            self.canvas.move(self.AutoUAV[i], xy_[i][0] - self.xy[i][0], xy_[i][1] - self.xy[i][1])
        self.xy = xy_

        # 状态空间归一化
        state_[:2] = (self.pxy - xy_).flatten() / 1000.  # 更新用户与无人机相对位置
        state_[2:4] = xy_.flatten() / 1000.  # 无人机绝对位置
        state_[4] = self.FX / 1000.  # 无人机越境次数/总步数
        state_[5] = self.Ec / 1000.
        Done = False

        # 奖励的定义——尽快到目的地/不要撞墙/减小能耗
        reward = -(abs(state_[0]) + abs(state_[1])) * 300 - self.FX * 20

        self.Q_dis()  # 获取所有用户与无人机的信道增益
        Dr = 0  # 无人机收集的数据量
        idu = 0
        T_hover = 0
        Ech = 168.49
        if self.E > self.E_need and gohome != 1 and self.remaining_time > self.t_need:
            if (above == False and self.dis[self.idx_target] <= self.sdc) or (
                    above == True and abs(state_[0]) <= 0.0008 and abs(state_[1]) <= 0.0008):
                Done = True
                reward += 1000
                # 只给目标用户收集数据
                data_rate = B * math.log(1 + P_u * self.h[self.idx_target] / self.noise, 2)
                need = P_u * (self.b_S[self.idx_target] / data_rate)  # 需要的能量
                T_hover1 = need / self.Non_linear_EH(P_d * self.h[self.idx_target])  # 充电时间
                T_hover2 = self.b_S[self.idx_target] / data_rate
                # self.remaining_time -= T_hover1
                E1 = (P_d + Ech) * T_hover1 + Ech * T_hover2
                if self.remaining_time - self.t_need > T_hover1 and (self.E - self.E_need) > E1:
                    self.remaining_time -= T_hover1
                    EcH = Ech * T_hover1 #充电时的悬停能耗
                    EcE = P_d * T_hover1  # 充电能耗：充电功率*充电时间
                    self.E -= (EcE + EcH)
                    T_hover2 = self.b_S[self.idx_target] / data_rate
                    E2 = Ech * T_hover2
                    if self.remaining_time - self.t_need > T_hover2 and (self.E - self.E_need) > E2:
                        EcH += Ech * T_hover2  # 数据收集时的悬停能耗
                        self.E -= Ech * T_hover2
                        T_hover = T_hover1 + T_hover2
                        Dr = self.b_S[self.idx_target]  # 收集全部的数据
                        self.b_S[self.idx_target] = 0
                        self.remaining_time -= T_hover2
                    elif (self.E - self.E_need) > E2:  # 时间不够
                        T_hover2 = self.remaining_time - self.t_need
                        self.E -= Ech * T_hover2
                        self.remaining_time = self.t_need
                        T_hover = T_hover1 + T_hover2
                        EcH += Ech * T_hover2
                        Dr = data_rate * T_hover2
                        self.b_S[self.idx_target] -= Dr
                    elif self.remaining_time - self.t_need > T_hover2:  # 能量不够
                        T_hover2 = (self.E - self.E_need) / Ech
                        self.remaining_time -= T_hover2
                        T_hover = T_hover1 + T_hover2
                        EcH += Ech * T_hover2
                        self.E -= self.E_need
                        Dr = data_rate * T_hover2
                        self.b_S[self.idx_target] -= Dr
                    idu = 1
                    self.Pass.append(self.pxy)
                else:  # 剩余时间或者剩余能量不够充电：不充电直接回家
                    # T_hover1 = self.remaining_time - self.t_need
                    # self.remaining_time = self.t_need
                    # EcH = Ech * T_hover1
                    # EcE = P_d * T_hover1  # 充电能耗：充电功率*充电时间
                    # T_hover = T_hover1
                    T_hover1 = 0
                    gohome = 1
        else:
            gohome = 1
        uav_consumed = EcE + EcF + EcH
        self.state = state_
        # EcE , EcF , EcH, uav_consumed
        return state_, reward, Done, Dr, EcE, EcF, EcH, ft, T_hover, idu, self.remaining_time - self.t_need, self.E, gohome  # 状态值，奖励，是否到达目标，总数据率, 覆盖到的用户，收集能量，无人机能耗

    # 每次无人机更新位置后，计算无人机与所有用户的距离与仰角，以及路径增益
    def Q_dis(self):
        for i in range(self.N_POI):
            self.dis[i] = math.sqrt(
                pow(self.SoPcenter[i][0] - self.xy[0][0], 2) + pow(self.SoPcenter[i][1] - self.xy[0][1], 2) + pow(
                    self.H, 2))  # 原始距离
            self.elevation[i] = 180 / math.pi * np.arcsin(self.H / self.dis[i])  # 仰角
            self.pro[i] = 1 / (1 + 10 * math.exp(-0.6 * (self.elevation[i] - 10)))  # 视距概率
            self.h[i] = (self.pro[i] + (1 - self.pro[i]) * 0.2) * pow(self.dis[i], -2.3) * pow(10,
                                                                                               -30 / 10)  # 参考距离增益为-30db

    # 输入是10-4W~10-5W,输出是0~9.079muW
    def Non_linear_EH(self, Pr):
        if Pr == 0:
            return 0
        P_prac = Mj / (1 + math.exp(-aj * (Pr - bj)))
        Peh = (P_prac - Mj * Oj) / (1 - Oj)  # 以W为单位
        return Peh

    # 输入是10-4W~10-5W,输出是0~9.079muW
    def linear_EH(self, Pr):
        if Pr == 0:
            return 0
        return Pr * pow(10, 6) * 0.2

    # 重选目标用户
    def CHOOSE_AIM(self):
        for i in range(len(self.Aim)):
            self.canvas.delete(self.Aim[i])
        # 重选目标用户
        self.now_id += 1
        self.idx_target = self.target[self.now_id]
        if self.idx_target < 100:
            self.pxy = self.SoPcenter[self.idx_target]  # 目标为传感器
        else:
            self.pxy = self.xy0[0]  # 目标为无人机原点
        L_AIM = self.canvas.create_rectangle(
            self.pxy[0] - 5, self.pxy[1] - 5,
            self.pxy[0] + 5, self.pxy[1] + 5,
            fill='red')
        self.Aim.append(L_AIM)

        self.state[:2] = (self.pxy - self.xy[0]).flatten() / 1000.
        self.render()
        return self.state

    # 调用Tkinter的update方法, 0.01秒去走一步。
    def render(self, t=0.01):
        time.sleep(t)
        self.update()

    def sample_action(self):
        v = np.random.rand()
        theta = -1 + 2 * np.random.rand()
        return [v, theta]

    def get_cache(self, coord):
        if isinstance(coord, np.ndarray):
            coord = tuple(coord)  # Convert to tuple if it's a numpy array
        elif isinstance(coord, (list, tuple)):
            coord = tuple(coord)  # Already a list or tuple, just convert to tuple
        else:
            coord = (coord,)  # Single value, convert to single-element tuple
        # 根据坐标获取ID（即索引）
        if coord in self.coord_to_id:
            return self.coord_to_id[coord]
        else:
            return None  # 如果坐标不在列表中，返回None或适当的错误处理


def update():
    for t in range(10):
        env.reset()
        while True:
            env.render()
            paras = env.sample_action()
            s_, r, done, dr, uav_consumed, ht, idu = env.step_move(paras)
            if done:
                break


if __name__ == '__main__':
    env = UAV()
    env.after(10, update)
    env.mainloop()

# 用于评估！好用