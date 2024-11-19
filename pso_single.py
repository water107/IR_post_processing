import os
import random
import shutil
import time
import numpy as np
from matplotlib import pyplot as plt
import subprocess

from DHas import DHas_similarity
from cos import cos_similarity
from malupdate import writer_to_txt

#
# input_dir = sys.argv[1]
#
ThRend_exe = "D:\\Study\\code_proj\\ThRend-master\\x64\\Release\\ThRend.exe"
index = 0
res = subprocess.Popen(ThRend_exe)
res.wait()

class PSO:

    def __init__(self, D, N, M, p_low, p_up, v_low, v_high, w=1., c1=2., c2=2.):
        self.w = w 
        self.c1 = c1  # 个体学习因子
        self.c2 = c2  # 群体学习因子
        self.D = D  
        self.N = N 
        self.M = M  
        self.p_range = [p_low, p_up]  # 粒子位置的约束范围
        self.v_range = [v_low, v_high]  # 粒子速度的约束范围
        self.x = np.zeros((self.N, self.D))  # 所有粒子的位置
        self.v = np.zeros((self.N, self.D))  # 所有粒子的速度
        self.p_best = np.zeros((self.N, self.D))  # 每个粒子的最优位置
        self.g_best = np.zeros((1, self.D))[0]  # 种群（全局）的最优位置
        self.p_bestFit = np.zeros(self.N)  # 每个粒子的最优适应值
        self.g_bestFit = float('Inf')  # float('-Inf')，始化种群（全局）的最优适应值，由于求极小值，故初始值给大，向下收敛，这里默认优化问题中只有一个全局最优解

        # 初始化所有个体和全局信息
        for i in range(self.N):
            for j in range(self.D):
                self.x[i][j] = random.uniform(self.p_range[0][j], self.p_range[1][j])
                self.v[i][j] = random.uniform(self.v_range[0], self.v_range[1])
            print(f"已初始化 {i+1} in {self.N} 个粒子")
            self.p_best[i] = self.x[i]  # 保存个体历史最优位置，初始默认第0代为最优
            fit = self.fitness(self.p_best[i])
            self.p_bestFit[i] = fit  # 保存个体历史最优适应值
            if fit < self.g_bestFit:  # 寻找并保存全局最优位置和适应值
                self.g_best = self.p_best[i]
                self.g_bestFit = fit

    def fitness(self, x):

        (x1, x2) = x
        writer_to_txt(x1, x2)
        res_in = subprocess.Popen([ThRend_exe])
        res_in.wait()
        image1 = 'apparent.png'
        image2 = 'ThRend.png'
        similarity = DHas_similarity(image1, image2)
        print('图片相似度：', similarity)
        return 1 - similarity

    def update(self, iter_n):
        for i in range(self.N):
            # 更新速度(核心公式)
            self.v[i] = self.w * self.v[i] + self.c1 * random.uniform(0, 1) * (
                    self.p_best[i] - self.x[i]) + self.c2 * random.uniform(0, 1) * (self.g_best - self.x[i])
            # 速度限制
            for j in range(self.D):
                if self.v[i][j] < self.v_range[0]:
                    self.v[i][j] = self.v_range[0]
                if self.v[i][j] > self.v_range[1]:
                    self.v[i][j] = self.v_range[1]
            # 更新位置
            self.x[i] = self.x[i] + self.v[i]
            # 位置限制
            for j in range(self.D):
                if self.x[i][j] < self.p_range[0][j]:
                    self.x[i][j] = self.p_range[0][j]
                if self.x[i][j] > self.p_range[1][j]:
                    self.x[i][j] = self.p_range[1][j]
            # 更新个体和全局历史最优位置及适应值
            _fit = self.fitness(self.x[i])
            print(f"已更新 {i+1} in {self.N} 个粒子, 第{iter_n}轮迭代")
            if _fit < self.p_bestFit[i]:
                self.p_best[i] = self.x[i]
                self.p_bestFit[i] = _fit
            if _fit < self.g_bestFit:
                self.g_best = self.x[i].copy()
                self.g_bestFit = _fit

    def pso(self, draw=1):
        best_fit = []  
        w_range = None
        if isinstance(self.w, tuple):
            w_range = self.w[1] - self.w[0]
            self.w = self.w[1]
        time_start = time.time() 
        for i in range(self.M):
            self.update(i)  
            if w_range:
                self.w -= w_range / self.M  
            print("\rIter: {:d}/{:d} fitness: {:.4f} ".format(i, self.M, self.g_bestFit, end='\n'))
            destination_path = os.path.join("D:\\Study\\code_proj\\ThRend-master\\outs\\iter", str(i+1) + ".png")
            try:
                shutil.copy("D:\\Study\\code_proj\\ThRend-master\\outs\\apparent.png", destination_path)
                print(f"成功复制并重命名图片为: {destination_path}")
            except Exception as e:
                print(f"复制或重命名文件时出错: {e}")
            best_fit.append(self.g_bestFit.copy())
        time_end = time.time()  
        print(f'Algorithm takes {time_end - time_start} seconds')  
        if draw:
            print(best_fit)
            plt.figure()
            plt.plot([i for i in range(self.M)], best_fit)
            plt.xlabel("iter")
            plt.ylabel("loss")
            plt.title("Iter process")
            plt.show()


if __name__ == '__main__':
    low = [0.5, 0.5]
    up = [1.0, 1.0]
    pso = PSO(2, 70, 20, low, up, -0.5, 0.5, w=0.9)
    pso.pso()
