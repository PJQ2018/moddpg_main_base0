import os
import time
from ga_env import UAV
from ddpg2 import AGENT
import datetime
import argparse
import math
import json
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.format': 'png',
    'savefig.bbox': 'tight',
    'figure.autolayout': True,
    'axes.unicode_minus': False,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9
})

parser = argparse.ArgumentParser(description='Evaluate the trained model.')
parser.add_argument('--is_train', type=int, default=1, metavar='train(1) or eval(0)',
                    help='train model of evaluate the trained model')

# TRAINING
parser.add_argument('--gamma', type=float, default=0.9, metavar='discount rate',
                    help='The discount rate of long-term returns')
parser.add_argument('--mem_size', type=int, default=8000, metavar='memorize size',
                    help='max size of the replay memory')
parser.add_argument('--batch_size', type=int, default=64, metavar='batch size',
                    help='batch size')
parser.add_argument('--lr_actor', type=float, default=0.0005, metavar='learning rate of actor',
                    help='learning rate of actor network')
parser.add_argument('--lr_critic', type=float, default=0.001, metavar='learning rate of critic',
                    help='learning rate of critic network')
parser.add_argument('--replace_tau', type=float, default=0.001, metavar='replace_tau',
                    help='soft replace_tau')
parser.add_argument('--Num_episode_plot', type=int, default=10, metavar='plot freq',
                    help='frequent of episodes to plot')
parser.add_argument('--save_model_freq', type=int, default=100, metavar='save freq',
                    help='frequent to save network parameters')
parser.add_argument('--R_dc', type=float, default=20., metavar='R_DC',
                    help='the radius of data collection')
parser.add_argument('--R_eh', type=float, default=20., metavar='R_EH',
                    help='the radius of energy harvesting')

# evaluation
parser.add_argument('--eval_num', type=int, default=3, metavar='EN',
                    help='number of episodes for evaluation')
parser.add_argument('--model', type=str, default='P_moddpg', metavar='model path',
                    help='the path of the trained model')
parser.add_argument('--save_path', type=str, default='eval_result', metavar='save path',
                    help='the save path of the evaluation result')
args = parser.parse_args()

#####################  set the save path  ####################
# base_path = os.path.join(os.getcwd(), args.save_path)
#
# for subfolder in ['logs', 'figs', 'drowing']:
#     full_path = os.path.join(base_path, subfolder)
#     os.makedirs(full_path, exist_ok=True)
logs_path = '/{}/{}/'.format(args.save_path, 'logs')
path = os.getcwd() + logs_path
if not os.path.exists(path):
    os.makedirs(path)
figs_path = '/{}/{}/'.format(args.save_path, 'figs')
path = os.getcwd() + figs_path
if not os.path.exists(path):
    os.makedirs(path)
os.makedirs(figs_path, exist_ok=True)

Q = 10.
V_me = 10.0
V_max = 35.0
EC_min = 126.
EC_hov = 168.49  # 悬停能耗
EC_grd = 506.69  # 178.28 356.26 506.69
yong = []
yong1 = []
yong2 = []

# 加载设备访问序列
def load_paths(filename="best_paths6.json"):
    with open(filename, 'r') as f:
        return json.load(f)

def evaluation(i):
    policy = comp_policies[i]
    load_ep = eps[i]
    if policy.find('P_Vmax') == -1 and policy.find('P_V_ME') == -1:
        ddpg = AGENT(args, a_num, a_dim, s_dim, a_bound, False)
        ddpg.load_ckpt(args.model, load_ep)
    file = open(os.path.join('.{}{}.txt'.format(logs_path, str(policy))), 'w+')
    file.write(
        'Episode|data rate|Harvasted energy|fly energy consumption|Total number of EH user|sum rate|Total number of ID user|Average harvasted energy|Average number of EH user' + '\n')
    file.flush()
    ep = 0
    while ep < args.eval_num:
        s = env.reset()
        Idu = 0
        FX = 0
        Dr = 0
        Ec = 0
        ep_reward = 0
        action = np.asarray([0., 0.])
        Ht = 0
        Ft = 0
        EcE = 0
        EcF = 0
        EcH = 0
        plot_MH = []  # Moddpg的悬停位置
        trajectory = []  # 初始化训练无人机轨迹记录列表
        while True:
            if policy.find('P_Vmax') != -1:
                count = "1"
                above = True
                if s[0] < 0:
                    action[0] = max(s[0] * V_max * 3 / math.sqrt(s[0] ** 2 + s[1] ** 2), s[0] * 1000) / (env.max_speed)
                else:
                    action[0] = min(s[0] * V_max * 3 / math.sqrt(s[0] ** 2 + s[1] ** 2), s[0] * 1000) / (env.max_speed)
                if s[1] < 0:
                    action[1] = max(s[1] * V_max * 3 / math.sqrt(s[0] ** 2 + s[1] ** 2), s[1] * 1000) / (env.max_speed) # * 3
                else:
                    action[1] = min(s[1] * V_max * 3 / math.sqrt(s[0] ** 2 + s[1] ** 2), s[1] * 1000) / (env.max_speed)
                s_, r, done, dr, Ece, Ecf, Ech, ft, ht, idu, rt, e, g= env.step_move(action, above, count)
            elif policy.find('P_V_ME') != -1:
                above = True
                count = "2"
                if s[0] < 0:
                    action[0] = max(s[0] * V_me / math.sqrt(s[0] ** 2 + s[1] ** 2), s[0] * 1000) / env.max_speed
                else:
                    action[0] = min(s[0] * V_me / math.sqrt(s[0] ** 2 + s[1] ** 2), s[0] * 1000) / env.max_speed
                if s[1] < 0:
                    action[1] = max(s[1] * V_me / math.sqrt(s[0] ** 2 + s[1] ** 2), s[1] * 1000) / env.max_speed
                else:
                    action[1] = min(s[1] * V_me / math.sqrt(s[0] ** 2 + s[1] ** 2), s[1] * 1000) / env.max_speed
                s_, r, done, dr, Ece, Ecf, Ech, ft, ht, idu, rt, e, g = env.step_move(action, above, count)
            else:
                action = ddpg.choose_action(s)
                s_, r, done, dr, Ece, Ecf, Ech, ft, ht, idu, rt, e, g = env.step_move(action)
                trajectory.append(s[2:4].copy() * 1000)

            # if ep == 0 and policy.find('P_moddpg') != -1:  #记录训练得到的无人机轨迹 and Idu > 65
            #     trajectory.append(s[2:4].copy() * 1000)

            Ft += ft
            FX += env.FX

            Ec += (Ece + Ecf + Ech)
            EcE += Ece
            EcF += Ecf
            EcH += Ech

            r += 0.001 * dr - (Ece + Ecf + Ech)
            # 2
            ep_reward += r
            if done:  # 悬停收集数据
                Idu += idu  # 服务用户计数
                Dr += dr
                Ht += ht
                s = env.CHOOSE_AIM()
                # 新增条件判断：当满足特定条件时记录悬停位置and Idu >= 66 and Idu >= 66 这是只记录后几个
                # if ep == 0 and policy.find('P_moddpg') != -1:
                #     hover_position = s[2:4].copy() * 1000  # 记录训练得到的无人机悬停位置
                #     plot_MH.append(hover_position)
            else:
                s = s_

            # if Idu >= (len(env.target) - 2):
            if rt < 1 or Idu >= (len(env.target) - 2) or (e <= env.E_need) or g == 1:
                trajectory.append(s[2:4].copy() * 1000)
                trajectory.append([0, 0])
                # (e <= env.E_need)
                Ft += env.t_need
                env.remaining_time -= env.t_need
                Ec += env.E_need
                EcF += env.E_need
                ep_reward -= env.E_need
                e -= env.E_need

                FX /= Ft #平均撞墙次数
                EEc = Ec / (Ht + Ft)  # 平均每步消耗能量

                ep += 1
                plot_x.append(ep)
                plot_Dr.append(Dr / 1000000.)  # 总收集数据量
                plot_Idu.append(Idu)  # 总服务设备个数
                plot_Ec.append(Ec)  # 总能耗
                plot_DE.append(Dr/Ec)  # 总能效
                plot_EcF.append(EcF)  # 总飞行能耗
                plot_EcE.append(EcE)  # 总充电能耗
                plot_EcH.append(EcH)  # 总悬停能耗
                plot_EEc.append(EEc)  # 平均能耗

                plot_HT.append(Ht)  # 悬停时间
                plot_FT.append(Ft)  # 飞行时间

                # 实时输出训练数据|adr:%i
                print(
                    'Episode:%i |Dr:%i |Idu:%i |D/E:%.2f |ep_reward:%.2f |EEc:%.2f |Ec:%.2f |EcE:%.2f |EcF:%.2f |EcH:%.2f |Ft:%.2f |Ht:%.2f' % (
                        ep, Dr, Idu, Dr/Ec, ep_reward, EEc, Ec, EcE, EcF, EcH, Ft, Ht))
                # 将相关数据写入文档
                write_str = '%i|%i|%i|%.3f|%.3f|%.3f|%i|%.3f|%.3f|%.2f|%.2f|%.2f\n' % (
                    ep, Dr, Idu, Dr/Ec, EEc, Ec, FX, EcE, EcF, EcH, Ft, Ht)

                # if policy.find('P_moddpg') != -1:
                #     data = load_paths()
                #     positions = np.vstack([env.SoPcenter, env.xy0])  # 包含所有设备和原点
                #     path_coords2 = np.vstack([env.xy0, trajectory, env.xy0])  # 优化后的路径
                #     path_coords = [positions[i] for i in env.target]  # 原始路径
                #     plt.figure(figsize=(10, 10))
                #
                #     # 定义颜色映射
                #     b_S = np.array(data["b_S"])  # 提取数据量
                #     unique_b_S = np.unique(b_S)  # 提取数据量  # 获取唯一的数据量值
                #     colors = ['#800080', '#FFA500', '#008000']
                #     for i, value in enumerate(unique_b_S):
                #         mask = (b_S == value)  # 找到当前数据量对应的设备
                #         plt.scatter(env.SoPcenter[mask, 0], env.SoPcenter[mask, 1],
                #                     c=[colors[i]], label=f'数据量 {value}')
                #
                #     for i, (x, y) in enumerate(env.SoPcenter):  # 设备旁边显示对应的序号
                #         plt.text(x, y, f'{i}', fontsize=12, ha='right')
                #
                #     plt.scatter(env.xy0[0][0], env.xy0[0][1], c='red', marker='*', s=200, label='原点')
                #
                #     for i in range(len(path_coords) - 1):  # 绘制带箭头的原路径
                #         x_start, y_start = path_coords[i]
                #         x_end, y_end = path_coords[i + 1]
                #         plt.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                #                      arrowprops=dict(arrowstyle='->', color='green', lw=1, linestyle='--'))
                #
                #     for i in range(len(path_coords2) - 1):   # 优化后的路径
                #         x_start, y_start = path_coords2[i]
                #         x_end, y_end = path_coords2[i + 1]
                #         plt.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                #                      arrowprops=dict(arrowstyle='-', color='red', lw=1.5, linestyle=':'))
                #
                #     # 6. 添加图例（手动补充路径标签）
                #     plt.plot([], [], 'g--', label='原始路径')
                #     plt.plot([], [], 'r-', label='优化路径')
                #
                #     plt.legend()
                #     plt.title('无人机路径规划（原始 vs 优化）')
                #     plt.xlabel('X')
                #     plt.ylabel('Y')
                #     # plt.savefig('DDPG UAV Trajectory Comparison.png')
                #     # plt.savefig('.{}DDPG UAV Trajectory Comparison.png'.format(drowing_path))
                #     plt.savefig(f'ep{ep}_DDPG_UAV_Trajectory.png'.format(drowing_path))
                #     plt.close()

                if policy.find('P_moddpg') != -1:
                    if Idu >= 74 and Ec <= 90500:
                        yong.append(ep - 1)
                    if Idu >= 74 and Ec <= 90600:
                        yong1.append(ep - 1)
                    if Idu >= 74 and Ec <= 90700:
                        yong2.append(ep - 1)

                # if ep == 1 and policy.find('P_moddpg') != -1:
                #     data = load_paths()
                #     positions = np.vstack([env.SoPcenter, env.xy0])  # 包含所有设备和原点
                #     path_coords = [positions[i] for i in env.target]  # 原始路径
                #     plt.figure(figsize=(3.5, 3), dpi=600)
                #
                #     # 定义颜色映射
                #     b_S = np.array(data["b_S"])  # 提取数据量
                #     unique_b_S = np.unique(b_S)   # 获取唯一的数据量值
                #     colors = ['#800080', '#FFA500', '#008000']
                #
                #     for i, value in enumerate(unique_b_S):
                #         mask = (b_S == value)  # 找到当前数据量对应的设备
                #         plt.scatter(env.SoPcenter[mask, 0], env.SoPcenter[mask, 1], s=6, alpha=0.8,
                #                     c=[colors[i]], label=f'Data volume {value}(Mb)')
                #     plt.scatter(env.xy0[0][0], env.xy0[0][1], c='cyan', marker='^', s=16, label='Origin')
                # #
                #     # 绘制原轨迹（蓝色虚线）
                #     # path_coords = np.array(path_coords)
                #     # plt.plot(path_coords[:, 0], path_coords[:, 1], color='blue', linestyle='--' , linewidth=2.0, alpha=0.9, label='P_Vmax')
                #
                #     # 绘制轨迹（红色虚线）
                #     trajectory = np.array(trajectory)
                #     plt.plot(trajectory[:, 0], trajectory[:, 1], color='red', linewidth=1.0, linestyle='-', alpha=0.9)
                #
                #     # 绘制悬停位置（红色星星）
                #     # for pos in plot_MH:
                #     #     plt.scatter(pos[0], pos[1], c='red', marker='*', s=49, alpha=0.5, edgecolors='black')
                #     #     # 绘制覆盖半径
                #     #     circle = plt.Circle((pos[0], pos[1]), 20, color='red', fill=True, alpha=0.4)
                #     #     plt.gca().add_patch(circle)
                #
                #     # 绘制正常悬停位置（蓝色星星）
                #     # for pos in path_coords[:]:
                #     #     plt.scatter(pos[0], pos[1], c='blue', marker='*', s=49, alpha=0.5, edgecolors='black')
                #     #     # 绘制覆盖半径
                #     #     circle = plt.Circle((pos[0], pos[1]), 20, color='blue', fill=True, alpha=0.4)
                #     #     plt.gca().add_patch(circle)
                #
                #     plt.xlabel('X (m)')
                #     plt.ylabel('Y (m)')
                #     plt.legend()
                #     plt.grid(linestyle='-.')
                #     # plt.title('UAV Hover Coverage')#f'{figs_path}{save_name}.png'
                #     plt.savefig(f'{figs_path}UAV Hover2 Coverage.png', bbox_inches='tight', pad_inches=0.1)
                #     plt.clf()
                #     # plt.close()

                def plot_uav_trajectory(ep, trajectory, prefix='UAV_Hover_Ep'):
                    data = load_paths()
                    # positions = np.vstack([env.SoPcenter, env.xy0])
                    # # path_coords = [positions[i] for i in env.target]

                    plt.figure(figsize=(3.5, 3), dpi=600)
                    # 设备数据量可视化
                    b_S = np.array(data["b_S"])
                    unique_b_S = np.unique(b_S)
                    colors = ['#800080', '#FFA500', '#008000']  # 按数据量分配颜色

                    for i, value in enumerate(unique_b_S):
                        mask = (b_S == value)
                        plt.scatter(env.SoPcenter[mask, 0], env.SoPcenter[mask, 1], s=2, alpha=0.8,
                                    c=[colors[i]], label=f'Data volume {value}(Mb)')

                    # 起点
                    plt.scatter(env.xy0[0][0], env.xy0[0][1], c='cyan', marker='^', s=9, label='Origin')

                    # UAV轨迹线
                    trajectory = np.array(trajectory)
                    plt.plot(trajectory[:, 0], trajectory[:, 1], color='red', linewidth=0.8, linestyle='-.', alpha=0.9,
                             label='Trajectory')

                    # 图形设置
                    plt.xlabel('X (m)')
                    plt.ylabel('Y (m)')
                    plt.legend(fontsize=8, loc='lower center')
                    plt.grid(linestyle='-.')

                    # 自动命名文件名（含回合号，补零对齐）
                    save_path = f'{figs_path}{prefix}{ep:03d}.png'
                    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
                    plt.clf()

                if policy.find('P_moddpg') != -1:
                    plot_uav_trajectory(ep, trajectory)
                file.write(write_str)
                file.flush()
                file.close
                break



if __name__ == '__main__':
    comp_policies = ['P_moddpg','P_Vmax', 'P_V_ME' ]
    #
    Label = [r'DODDPG', r'MVHS', r'MEHS']
    eps = [2000, 0, 0]

    t1 = time.time()
    plot_x_avg = []
    plot_Dr_avg = []  # 总收集数据量
    plot_Idu_avg = []  # 总服务设备个数
    plot_DE_avg = []  # 总能耗
    plot_Ec_avg = []  # 总能耗
    plot_EEc_avg = []  # 平均能耗
    plot_EcF_avg = []  # 总飞行能耗
    plot_EcE_avg = []  # 总充电能耗
    plot_EcH_avg = []  # 总悬停能耗
    plot_HT_avg = []  # 悬停时间
    plot_FT_avg = []  # 飞行时间

    for i in range(len(comp_policies)):

        env = UAV()
        print(comp_policies[i])
        np.random.seed(1)

        # 定义状态空间，动作空间，动作幅度范围
        s_dim = env.state_dim
        a_num = 2
        a_dim = env.action_space.shape[0]
        a_bound = env.action_space

        plot_x = []
        plot_Dr = []  # 总收集数据量
        plot_DE = []  #数据量/能耗 = 能效
        plot_Idu = []  # 总服务设备个数
        plot_Ec = []  # 总能耗
        plot_EEc = []  # 平均能耗
        plot_EcF = []  # 总飞行能耗
        plot_EcE = []  # 总充电能耗
        plot_EcH = []  # 总悬停能耗
        # plot_MH = [] # Moddpg的悬停位置
        # plot_VH = []  # Vmax的悬停位置
        plot_HT = []  # 悬停时间
        plot_FT = []  # 飞行时间
        # trajectory = []  # 初始化训练无人机轨迹记录列表

        evaluation(i)

        plot_x_avg.append(plot_x)
        plot_Dr_avg.append(plot_Dr) # 总收集数据量
        plot_Idu_avg.append(plot_Idu)   # 总服务设备个数
        plot_DE_avg.append(plot_DE)  # 总能耗
        plot_Ec_avg.append(plot_Ec)  # 总能耗
        plot_EEc_avg.append(plot_EEc)   # 平均能耗
        plot_EcF_avg.append(plot_EcF)  # 总飞行能耗
        plot_EcE_avg.append(plot_EcE)  # 总充电能耗
        plot_EcH_avg.append(plot_EcH)  # 总悬停能耗
        plot_HT_avg.append(plot_HT) # 悬停时间
        plot_FT_avg.append(plot_FT)  # 飞行时间

    '''
        # 画图
        1、累积奖励Accumulated reward，2、回合收集数据量 sum rate 3、 平均每次悬停收集数据量 data rate
        4、 回合总收集能量harvested energy  5、平均每次悬停收集能量Average harvested energy
        6、回合平均每步飞行能耗 fly energy consumption  7、上传数据用户数 The number of ID user 
        8、总充电用户数 The number of EH user 9、平均每次悬停充电用户数 Average number of EH user 
        10、系统平均数据水平 Average data buffer length 11、 数据溢出用户数 N_d
    '''
    # Fig 1:Average data rate(sum_rate/idu)/Total harvested energy(Ec)/Average flying energy consumption(Ec/Ft)
    # 论文图五：平均数据速率（sum_rate/idu）/ 总捕获能量（Ec）/ 平均飞行能量消耗（Ec/Ft）

    # Fig 2:Total number of DC devices(idu)/Average energy harvesting rate(Ec/Ht)/Average number of EH devices(ehu/idu)
    ############################################data_rate/harvested energy########################
    ####################################fly energy consumption/total number of EH user########
    # 论文图 每次悬停平均数据率、# 总收集能量、平均能耗、总充电用户数、回合收集数据量、上传数据用户数、平均每次悬停充电用户数、
    print("len(yong) yong", len(yong), yong)
    print("len(yong) yong1", len(yong1), yong1)
    print("len(yong) yong2", len(yong2), yong2)

    font_label = {'fontsize': 9}
    figsize = (7, 3)  # 3.5 英寸 * 2 = 7 英寸宽，适合双图排列

    def plot_two_subplots(x_data, y_data1, y_data2, y_label1, y_label2, title1, title2, save_name):
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        for i in range(len(x_data)):
            axes[0].plot(x_data[i], y_data1[i], linewidth=1.5, markersize=5,
                         label=f'{Label[i]}: {np.mean(y_data1[i]):.2f}')
            axes[1].plot(x_data[i], y_data2[i], linewidth=1.5, markersize=5,
                         label=f'{Label[i]}: {np.mean(y_data2[i]):.2f}')
        for ax, ylabel, title in zip(axes, [y_label1, y_label2], [title1, title2]):
            ax.set_xlabel('Episodes', font_label)
            ax.set_ylabel(ylabel, font_label)
            ax.set_title(title, pad=8)
            ax.grid(linestyle='-.')
            ax.legend(fontsize=8, loc='center right', framealpha=0.5)
        plt.subplots_adjust(wspace=0.3, left=0.08, right=0.96)

        plt.savefig(f'{figs_path}{save_name}.png')
        # plt.savefig(f'{figs_path}{save_name}.eps', format='eps')
        print(f'{figs_path}{save_name}')
        # plt.clf()  marker='.',

    # 图1：数据量 & 总能耗
    plot_two_subplots(plot_x_avg, plot_Dr_avg, plot_Ec_avg,
                      'Data Volume (Mb)', 'Total Energy Consumption (W)',
                      '(a)', '(b)', 'fig1_combined')

    # 图2：能效 & 平均能耗
    plot_two_subplots(plot_x_avg, plot_DE_avg, plot_EEc_avg,
                      'Energy Efficiency (Mb/W)', 'Average Energy Consumption (W/s)',
                      '(a)', '(b)', 'fig2_combined')

    # 图3：飞行 & 悬停能耗
    plot_two_subplots(plot_x_avg, plot_EcF_avg, plot_EcH_avg,
                      'Flight Energy Consumption (W)', 'Hover Energy Consumption (W)',
                      '(a)', '(b)', 'fig3_combined')

    # 图4：飞行时间 & 悬停时间
    plot_two_subplots(plot_x_avg, plot_HT_avg, plot_FT_avg,
                      'Hovering Time (s)', 'Flying Time (s)',
                      '(a)', '(b)', 'time')
    #########################################################################

    now_time = datetime.datetime.now()
    date = now_time.strftime('%Y-%m-%d %H_%M_%S')
    print('Running time: ', time.time() - t1)
