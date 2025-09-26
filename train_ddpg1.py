import os
import time
from env0 import UAV
from ddpg1 import AGENT
import datetime
import matplotlib.pyplot as plt
import numpy as np
import argparse
import tensorflow as tf
import math
# 以上是运行代码需要的文件

# 这是画图参数的定义
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

# 这一大段是数据参数的定义
parser = argparse.ArgumentParser(description='Train the DDPG model.')
parser.add_argument('--is_train', type=int, default=1, metavar='train(1) or eval(0)',
                    help='train model of evaluate the trained model')
# TRAINING
parser.add_argument('--gamma', type=float, default=0.9, metavar='discount rate',
                    help='The discount rate of long-term returns')
parser.add_argument('--mem_size', type=int, default=8000, metavar='memorize size',
                    help='max size of the replay memory')
parser.add_argument('--batch_size', type=int, default=64, metavar='batch size',
                    help='batch size')
parser.add_argument('--lr_actor', type=float, default=0.001, metavar='learning rate of actor',
                    help='learning rate of actor network')
parser.add_argument('--lr_critic', type=float, default=0.001, metavar='learning rate of critic',
                    help='learning rate of critic network')
parser.add_argument('--replace_tau', type=float, default=0.001, metavar='replace_tau',
                    help='soft replace_tau')
parser.add_argument('--episode_num', type=int, default=2001, metavar='episode number',
                    help='number of episodes for training')
parser.add_argument('--Num_episode_plot', type=int, default=10, metavar='plot freq',
                    help='frequent of episodes to plot')
parser.add_argument('--save_model_freq', type=int, default=100, metavar='save freq',
                    help='frequent to save network parameters')
parser.add_argument('--model', type=str, default='P_moddpg', metavar='save path',
                    help='the save path of the train model')
parser.add_argument('--R_dc', type=float, default=20., metavar='R_DC',
                    help='the radius of data collection')
parser.add_argument('--R_eh', type=float, default=20., metavar='R_EH',
                    help='the radius of energy harvesting')

args = parser.parse_args()

#####################  set the save path 图片的保存路径 ####################
model_path = '/{}/{}/'.format(args.model, 'models1')
path = os.getcwd() + model_path
if not os.path.exists(path):
    os.makedirs(path)
logs_path = '/{}/{}/'.format(args.model, 'logs')
path = os.getcwd() + logs_path
if not os.path.exists(path):
    os.makedirs(path)

figs_path = os.path.join(os.getcwd(), args.model, 'figs')
os.makedirs(figs_path, exist_ok=True)

# figs_path = '/{}/{}/'.format(args.model, 'figs')
# path = os.getcwd() + figs_path
# if not os.path.exists(path):
#     os.makedirs(path)
# os.makedirs(figs_path, exist_ok=True)

# 设置画图横纵坐标字体格式
font1 = {'family': 'Times New Roman', 'size': 9}
font2 = {'family': 'Times New Roman'}
# 全局参数
Q = 10.
V_ME = 10. # 最省电的速度
V_max = 35.0 #定义的最大速度
EC_min = 126. #最省电速度时的能耗
EC_hov = 168.49  # 悬停时能耗
EC_grd = 506.69 # 178. 最大速度的能耗
epsilon = 1e-8

def train():
    var = 2.  # control exploration 探索参数
    for ep in range(args.episode_num):
        # if ep == 16:
        #     print("ep", ep)
        s = env.reset() #环境初始化
        ep_reward = 0 #回合奖励
        Idu = 0  # 服务用户计数
        Dr = 0  # 总数据量
        Ec = 0  # 总能耗
        TD_error = 0  # critic网络训练误差
        A_loss = 0  # actor网络Q值
        Ht = 0  # 记录悬停时间
        Ft = 0  # 记录飞行时间
        Fx = 0  #记录撞墙次数
        update_network = 0  # 记录网络训练次数
        while True:
            ft = 1  # 飞行时间
            act = agent.choose_action(s) #根据状态选择动作
            act = np.clip(np.random.normal(act, var), a_bound.low,
                          a_bound.high)  # add randomness to action selection for exploration 增大探索空间
            s_, r, done, dr, Ece, Ecf, Ech, ht, idu, rt = env.step_move(act) #将动作应用到环境中，获取下一状态、奖励值、以及各项参数
            # 下一状态、辅助奖励、是否在正上方（这里没用，评估时使用的另一个环境文件、数据收集量、充电、飞行、悬停参数、悬停时间、服务设备个数、剩余可用时间）
            Ft += ft #飞行时间累加
            Ec += (Ece + Ecf + Ech) #能耗累加
            Fx += env.FX #撞墙次数累加
            Dr += dr #数据收集量累加

            r += 0.001 * dr - (Ece + Ecf + Ech) #奖励
            # r += (Dr + epsilon) / Ec
            ep_reward += r
            agent.store_transition(s, act, r, s_)
            if agent.pointer > args.mem_size:
                var = max([var * 0.999, 0.1])
                td_error, a_loss = agent.learn()
                update_network += 1
                TD_error += td_error
                A_loss += a_loss

            if done:  # 目标设备落入无人机DC范围内
                Idu += idu  # 服务用户计数
                Ht += ht
                s = env.CHOOSE_AIM()
            else:
                s = s_
            ec = EC_min * env.t_need
            if Idu >= len(env.target)- 2 or rt < 1:
                # if var <=0.11:  or
                #     print("var",var)
                # :
                Ft += env.t_need
                # ep_reward -= 2 * ec
                Ec += ec

                env.remaining_time -= env.t_need
                # if env.remaining_time < 0:
                #     ep_reward -= 3000

                if update_network != 0:
                    TD_error /= update_network
                    A_loss /= update_network

                # FX /= Ft
                EEc = Ec / (Ft + Ht)  # 平均能耗
                if Idu:
                    adr = Dr / Idu  # 平均数据量
                else:
                    adr = 0


                print(
                    'Ep:%i |TD_error:%i |A_loss:%i |ep_r:%i |Idu:%i |Dr:%i |Ec:%.2f |Ft:%i |Ht:%.2f |Fx:%i' % (
                        ep, TD_error, A_loss, ep_reward, Idu, Dr, Ec, Ft, Ht, Fx))


                '''
                # 将相关数据写入文档
                # Fig 1:Average data rate(sum_rate/idu)/Total harvested energy(Ec)/Average flying energy consumption(Ec)
                # Fig 2:Total number of DC devices(idu)/Average energy harvesting rate(Ec/Ht)/Average number of EH devices(ehu/idu)
                '''
                write_str = '%i|%.3f|%.3f|%.3f|%.3f|%i|%.3f|%.3f|%i|%i|%i\n' % (
                    ep, ep_reward, TD_error, A_loss, Dr, Idu, adr, Ec, Ft, Ht, Fx)
                file.write(write_str)
                file.flush()

                # 将相关数据存入队列用以画图
                plot_x.append(ep)
                plot_TD_error.append(TD_error)
                plot_A_loss.append(A_loss)
                plot_R.append(ep_reward)  # 累计奖励
                plot_Dr.append(Dr / 1000000)  # 回合总悬停吞吐量
                plot_Idu.append(Idu)  # 回合总收集数据用户
                plot_adr.append(adr)  # 回合平均数据量
                plot_Ec.append(Ec)  # 回合平均能耗
                plot_HT.append(Ht)  # 悬停时间
                plot_FT.append(Ft)  # 飞行时间
                break

        if ep % args.save_model_freq == 0 and ep != 0:
            agent.save_ckpt(model_path, ep)


if __name__ == '__main__':

    # 初始化环境
    env = UAV(args.R_dc)

    # reproducible，设置随机种子，为了能够重现
    env.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)

    # 定义状态空间，动作空间，动作幅度范围
    s_dim = env.state_dim
    a_num = 2
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space
    '''
    print('s_dim',s_dim)
    print('a_dim',a_dim)
    print('r_dim',r_dim)
    '''
    # 用agent算法
    agent = AGENT(args, a_num, a_dim, s_dim, a_bound, True)

    # 训练部分：
    t1 = time.time()
    plot_x = []
    plot_TD_error = []
    plot_A_loss = []
    plot_R = []  # 每回合累计奖励
    plot_Dr = []  # 每回合总悬停吞吐量
    plot_Idu = []  # 回合总收集数据用户
    plot_adr = []  # 回合平均数据量
    plot_Ec = []  # 每回合能耗（飞行或悬停）
    plot_Ec_eh = []  # 每回合能耗（为设备充充电）
    plot_HT = []  # 悬停时间
    plot_FT = []  # 飞行时间

    file = open(os.path.join('.{}{}'.format(logs_path, 'log1.txt')), 'w+')
    train()
    file.close

    #######################################1、累积奖励Accumulated reward##############################################
    # 单图
    def plot_single_curve(x, y, xlabel, ylabel, filename, figsize=(3.5, 3)):  # figsize in inches
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, y, linewidth=1.5)
        ax.grid(linestyle='-.')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.savefig(f'{figs_path}/{filename}', bbox_inches='tight')
        # plt.show()


    # 多图
    def plot_multi_curves(data_list, xlabel_list, ylabel_list, titles, filename, layout=(2, 1), figsize=(3.5, 6)):
        fig, axs = plt.subplots(*layout, figsize=figsize)
        axs = axs.flatten()
        for i, (x, y) in enumerate(data_list):
            axs[i].plot(x, y, linewidth=1.5)
            axs[i].grid(linestyle='-.')
            axs[i].set_xlabel(xlabel_list[i])
            axs[i].set_ylabel(ylabel_list[i])
            axs[i].set_title(titles[i], pad=10)
        plt.subplots_adjust(wspace=0.3)
        plt.savefig(f'{figs_path}/{filename}', bbox_inches='tight')
        # plt.close()


    # 单图
    plot_single_curve(plot_x, plot_R, 'Training Episodes', 'Accumulated Reward', 'Reward1.png')
    plot_single_curve(plot_x, plot_TD_error, 'Training Episodes', 'TD error of critic', 'TD error of critic1.png')
    plot_single_curve(plot_x, plot_A_loss, 'Training Episodes', 'Loss of actor', 'Loss of Actor1.png')

    # 双图
    plot_multi_curves(
        [(plot_x, plot_R), (plot_x, plot_TD_error)],
        ['Training Episodes', 'Training Episodes'],
        ['Accumulated Reward', 'TD error of critic'],
        ['(a)', '(b)'],
        'combine reward error1.png',
        layout=(1, 2),
        figsize=(7, 3)
    )

    plot_multi_curves(
        [(plot_x, plot_R), (plot_x, plot_A_loss)],
        ['Training Episodes', 'Training Episodes'],
        ['Accumulated Reward', 'Loss of actor'],
        ['(a)', '(b)'],
        'combine reward loss1.png',
        layout=(1, 2),
        figsize=(7, 3)
    )

    # 三图
    plot_multi_curves(
        [(plot_x, plot_R), (plot_x, plot_A_loss), (plot_x, plot_TD_error)],
        ['Training Episodes'] * 3,
        ['Accumulated Reward', 'Loss of actor', 'TD error of critic'],
        ['(a)', '(b)', '(c)'],
        'combine reward loss error1.png',
        layout=(1, 3),
        figsize=(10.5, 3)
    )

    x = 0
    plot_sr1 = 0  # 回合收集数据量
    plot_Ec1 = 0  # 平均能耗
    # # 奖励函数
    # fig =plt.figure(figsize=(3.5, 3))
    # ax = fig.add_subplot(1, 1, 1)  # R
    # ax.tick_params(labelsize=9)
    # ax.grid(linestyle='-.')
    # ax.plot(plot_x, plot_R)
    #
    # ax.set_xlabel('Training Episodes', font1)
    # ax.set_ylabel('Accumulated Reward', font1)
    #
    # fig.tight_layout()
    # plt.savefig('.{}{}'.format(figs_path, 'Reward1.png'), bbox_inches='tight')
    #
    # # 梯度误差
    # fig = plt.figure(figsize=(3.5, 3))
    # ax = fig.add_subplot(1, 1, 1)  # R
    # ax.tick_params(labelsize=9)
    # ax.grid(linestyle='-.')
    # ax.plot(plot_x, plot_TD_error)
    # ax.set_xlabel('Training Episodes', font1)
    # ax.set_ylabel('TD error of critic', font1)
    # label1 = ax.get_xticklabels() + ax.get_yticklabels()
    # [label.set_fontname('Times New Roman') for label in label1]
    # fig.tight_layout()
    # plt.savefig('.{}{}'.format(figs_path, 'TD error of critic1.png'), bbox_inches='tight')
    #
    # # 损失函数
    # fig = plt.figure(figsize=(3.5, 3))
    # ax = fig.add_subplot(1, 1, 1)  # R
    # ax.tick_params(labelsize=9)
    # ax.grid(linestyle='-.')
    # ax.plot(plot_x, plot_A_loss)
    # ax.set_xlabel('Training Episodes', font1)
    # ax.set_ylabel('Loss of actor', font1)
    # label1 = ax.get_xticklabels() + ax.get_yticklabels()
    # [label.set_fontname('Times New Roman') for label in label1]
    # fig.tight_layout()
    # plt.savefig('.{}{}'.format(figs_path, 'Loss of Actor1.png'), bbox_inches='tight')
    #
    #
    # #######################################1、loss##############################################
    # # 画图 奖励 梯度误差
    # fig = plt.figure(figsize=(7.15, 3))
    #
    # ax1 = fig.add_subplot(1, 2, 1)
    # ax2 = fig.add_subplot(1, 2, 2)
    # for ax in [ax1, ax2]:
    #     ax.grid(linestyle='-.')
    #
    # ax1.plot(plot_x, plot_R)
    # ax1.set_xlabel('Training Episodes')
    # ax1.set_ylabel('Accumulated reward')
    # ax1.set_title('(a)', pad=15)
    #
    # ax2.plot(plot_x, plot_TD_error)
    # ax2.set_xlabel('Training Episodes')
    # ax2.set_ylabel('TD error of critic')
    # ax2.set_title('(b)', pad=15)
    #
    # plt.subplots_adjust(wspace=0.3)
    # plt.savefig('.{}{}'.format(figs_path, 'combine reward error1.png'), bbox_inches='tight')
    #
    # # 画图 奖励 损失函数
    # fig = plt.figure(figsize=(16, 8))
    # ax1 = fig.add_subplot(1, 2, 1)
    # ax1.tick_params(labelsize=12)
    # ax1.grid(linestyle='-.')
    # ax2 = fig.add_subplot(1, 2, 2)
    # ax2.tick_params(labelsize=12)
    # ax2.grid(linestyle='-.')
    #
    # ax1.plot(plot_x, plot_R)
    # ax1.set_xlabel('Training Episodes', font1)
    # ax1.set_ylabel('Accumulated reward', font1)
    # ax1.set_title('(a)', fontdict=font1, pad=15)
    # label1 = ax1.get_xticklabels() + ax1.get_yticklabels()
    # [label.set_fontname('Times New Roman') for label in label1]
    #
    # ax2.plot(plot_x, plot_A_loss)
    # ax2.set_xlabel('Training Episodes', font1)
    # ax2.set_ylabel('Loss of actor', font1)
    # ax2.set_title('(b)', fontdict=font1, pad=15)
    # label2 = ax2.get_xticklabels() + ax2.get_yticklabels()
    # [label.set_fontname('Times New Roman') for label in label2]
    #
    # plt.subplots_adjust(wspace=0.3)
    # plt.savefig('.{}{}'.format(figs_path, 'combine reward loss1.png'), bbox_inches='tight')
    #
    # # 画图 奖励 梯度误差 损失函数
    # fig = plt.figure(figsize=(24, 8))
    # ax1 = fig.add_subplot(1, 3, 1)
    # ax1.tick_params(labelsize=12)
    # ax1.grid(linestyle='-.')
    # ax2 = fig.add_subplot(1, 3, 2)
    # ax2.tick_params(labelsize=12)
    # ax2.grid(linestyle='-.')
    # ax3 = fig.add_subplot(1, 3, 3)
    # ax3.tick_params(labelsize=12)
    # ax3.grid(linestyle='-.')
    #
    # ax1.plot(plot_x, plot_R)
    # ax1.set_xlabel('Training Episodes', font1)
    # ax1.set_ylabel('Accumulated reward', font1)
    # ax1.set_title('(a)', fontdict=font1, pad=15)
    # label1 = ax1.get_xticklabels() + ax1.get_yticklabels()
    # [label.set_fontname('Times New Roman') for label in label1]
    #
    # ax2.plot(plot_x, plot_A_loss)
    # ax2.set_xlabel('Training Episodes', font1)
    # ax2.set_ylabel('Loss of actor', font1)
    # ax2.set_title('(b)', fontdict=font1, pad=15)
    # label2 = ax2.get_xticklabels() + ax2.get_yticklabels()
    # [label.set_fontname('Times New Roman') for label in label2]
    #
    # ax3.plot(plot_x, plot_TD_error)
    # ax3.set_xlabel('Training Episodes', font1)
    # ax3.set_ylabel('TD error of critic', font1)
    # ax3.set_title('(c)', fontdict=font1, pad=15)
    # label3 = ax3.get_xticklabels() + ax3.get_yticklabels()
    # [label.set_fontname('Times New Roman') for label in label3]
    #
    # plt.subplots_adjust(wspace=0.3)
    # plt.savefig('.{}{}'.format(figs_path, 'combine reward loss error1.png'), bbox_inches='tight')
    #
    # ##################取10个画图##################################
    # x = 0
    # plot_sr1 = 0  # 回合收集数据量
    # plot_Ec1 = 0  # 平均能耗
    # plot_idu1 = 0  # 上传数据用户数
    #
    # plot_x_avg = []
    # plot_sr_avg = []  # 回合收集数据量
    # plot_Ec_avg = []  # 平均能耗
    # plot_idu_avg = []  # 上传数据用户数
    #
    # for i in range(1, len(plot_x)):
    #     x += i
    #     plot_sr1 += plot_Dr[i]  # 回合收集数据量
    #     plot_Ec1 += plot_Ec[i]  # 平均能耗
    #     plot_idu1 += plot_Idu[i]  # 上传数据用户数
    #
    #     if i % args.Num_episode_plot == 0 and i != 0:
    #         plot_x_avg.append(x / args.Num_episode_plot)
    #         plot_sr_avg.append(plot_sr1 / args.Num_episode_plot)  # 回合收集数据量
    #         plot_Ec_avg.append(plot_Ec1 / args.Num_episode_plot)  # 平均能耗
    #         plot_idu_avg.append(plot_idu1 / args.Num_episode_plot)  # 上传数据用户数
    #
    #         x = 0
    #         plot_sr1 = 0  # 回合收集数据量
    #         plot_Ec1 = 0  # 平均能耗
    #         plot_idu1 = 0  # 上传数据用户数
    #         plot_DQ1 = 0  # 平均用户数据缓存量
    #
    #     #####################################################################
    #     '''
    #     # 画图
    #     1、累积奖励Accumulated reward，2、回合收集数据量 sum rate 3、 平均每次悬停收集数据量 data rate
    #     4、 回合总收集能量harvested energy  5、平均每次悬停收集能量Average harvested energy
    #     6、回合平均每步飞行能耗 fly energy consumption  7、上传数据用户数 The number of ID user
    #     8、总充电用户数 The number of EH user 9、平均每次悬停充电用户数 Average number of EH user
    #     10、系统平均数据水平 Average data buffer length 11、 数据溢出用户数 N_d
    #     '''
    # # Fig 1:Average data rate(sum_rate/idu)/Total harvested energy(Ec)/Average flying energy consumption(Ec/Ft)
    # # Fig 2:Total number of DC devices(idu)/Average energy harvesting rate(Ec/Ht)/Average number of EH devices(ehu/idu)
    # ############################################main_result_1########################
    # # 图5 训练曲线跟踪优化目标：（a）总数据量；（c）平均能量消耗。
    # p1 = plt.figure(figsize=(16, 8))  # 第一幅子图,并确定画布大小
    # ax1 = p1.add_subplot(1, 2, 1)
    # ax1.tick_params(labelsize=12)
    # ax1.grid(linestyle='-.')
    # ax3 = p1.add_subplot(1, 2, 2)
    # ax3.tick_params(labelsize=12)
    # ax3.grid(linestyle='-.')
    #
    # ax1.plot(plot_x, plot_Dr)
    # ax1.set_xlabel('Training Episodes', font1)
    # ax1.set_ylabel('Sum data MB', font1)
    # ax1.set_title('(a)', fontdict=font1, pad=15)
    # label1 = ax1.get_xticklabels() + ax1.get_yticklabels()
    # [label.set_fontname('Times New Roman') for label in label1]
    #
    # ax3.plot(plot_x_avg, plot_Ec)
    # ax3.set_xlabel('Training Episodes', font1)
    # ax3.set_ylabel('Energy consumption (W)', font1)
    # ax3.set_title('(b)', fontdict=font1, pad=15)
    # label3 = ax3.get_xticklabels() + ax3.get_yticklabels()
    # [label.set_fontname('Times New Roman') for label in label3]
    #
    # plt.subplots_adjust(wspace=0.3)
    # plt.savefig('.{}{}'.format(figs_path, 'FIG_5.png'), bbox_inches='tight')
    #
    # ####################################################################################################################
    # # Fig 2:Total number of DC devices(idu)/Average energy harvesting rate(Ec/Ht)/Average number of EH devices(ehu/idu)
    #
    # ############################################10、系统平均数据水平 Average data buffer length#####################################################
    # # 平均用户数据缓冲量、数据溢出用户计数

    ###################################################################################
    now_time = datetime.datetime.now()
    date = now_time.strftime('%Y-%m-%d %H_%M_%S')
    print('Running time: ', time.time() - t1)
