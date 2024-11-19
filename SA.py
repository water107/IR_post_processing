import subprocess
import math
import random
import time
import matplotlib.pyplot as plt

from DHas import DHas_similarity
from cos import cos_similarity
from malupdate import writer_to_txt

ThRend_exe = "D:\\Study\\code_proj\\ThRend-master\\x64\\Release\\ThRend.exe"

def fitness(x1, x2):

    writer_to_txt(x1, x2)
    res_in = subprocess.Popen([ThRend_exe])
    res_in.wait()
    # col_image = cv2.imread("apparent.png", cv2.IMREAD_GRAYSCALE)
    # cv2.imwrite("r3.png",col_image)
    image1 = 'apparent.png'
    image2 = 'ThRend.png'
    similarity = DHas_similarity(image1, image2)
    print('图片相似度：', similarity)
    return 1 - similarity


class SA:
    def __init__(self, func, iter=100, T0=100, Tf=0.01, alpha=0.99):
        self.func = func
        self.iter = iter         #内循环迭代次数,即为L =100
        self.alpha = alpha       #降温系数，alpha=0.99
        self.T0 = T0             #初始温度T0为100
        self.Tf = Tf             #温度终值Tf为0.01
        self.T = T0              #当前温度
        # self.x = [random() * 11 -5  for i in range(iter)] #随机生成100个x的值
        # self.y = [random() * 11 -5  for i in range(iter)] #随机生成100个y的值
        self.x = [random.uniform(0, 1) for _ in range(iter)]
        self.y = [random.uniform(0, 1) for _ in range(iter)]
        self.most_best =[]
        self.history = {'f': [], 'T': []}

    def generate_new(self, x, y):   #扰动产生新解的过程
        while True:
            x_new = x + self.T * (random.random() - random.random())
            y_new = y + self.T * (random.random() - random.random())
            if (0 <= x_new <= 1) & (0 <= y_new <= 1):  
                break                                  #重复得到新解，直到产生的新解满足约束条件
        return x_new, y_new 

    def Metrospolis(self, f, f_new):   #Metropolis准则
        if f_new <= f:
            return 1
        else:
            p = math.exp((f - f_new) / self.T)
            if random.random() < p:
                return 1
            else:
                return 0

    def best(self):    #获取最优目标函数值
        f_list = []    #f_list数组保存每次迭代之后的值
        for i in range(self.iter):
            f = self.func(self.x[i], self.y[i])
            f_list.append(f)
        f_best = min(f_list)
        
        idx = f_list.index(f_best)
        return f_best, idx    #f_best,idx分别为在该温度下，迭代L次之后目标函数的最优解和最优解的下标

    def run(self):
        count = 0
        #外循环迭代，当前温度小于终止温度的阈值
        while self.T > self.Tf:       
                  
            #内循环迭代100次
            for i in range(self.iter): 
                f = self.func(self.x[i], self.y[i])                    #f为迭代一次后的值
                x_new, y_new = self.generate_new(self.x[i], self.y[i]) #产生新解
                f_new = self.func(x_new, y_new)                        #产生新值
                if self.Metrospolis(f, f_new):                         #判断是否接受新值
                    self.x[i] = x_new             #如果接受新值，则把新值的x,y存入x数组和y数组
                    self.y[i] = y_new
                print(f'第{count + 1}次迭代,第{i + 1}次循环,x1={self.x[i]},x2={self.y[i]}')
            # 迭代L次记录在该温度下最优解
            ft, _ = self.best()
            self.history['f'].append(ft)
            self.history['T'].append(self.T)
            #温度按照一定的比例下降（冷却）
            self.T = self.T * self.alpha        
            count += 1
            # 得到最优解
        f_best, idx = self.best()
        print(f"F={f_best}, x={self.x[idx]}, y={self.y[idx]}")

if __name__ == '__main__':
    time_start = time.time()
    sa = SA(fitness) 
    sa.run()
    plt.plot(sa.history['T'], sa.history['f'])
    plt.title('SA')
    plt.xlabel('T')
    plt.ylabel('f')
    plt.gca().invert_xaxis()
    plt.show()
    time_end = time.time()
    print(f'本次迭代耗时 {time_end - time_start} 秒')
