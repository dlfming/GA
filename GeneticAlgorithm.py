from operator import ne
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import os

class GeneticAlgorithm:
    def __init__(self, number,rounds):
        self.matrix = None
        self.good = None
        self.bad = None
        self.solutions = []
        self.remains = []
        self.rounds = rounds
        self.number = number
        self.iter = 0

    # 构建随机矩阵
    def init_random_matrix(self):
        # 生成随机矩阵
        matrix = np.random.randint(0,256,size=[64, 64])
        matrix[matrix<2] = 0
        matrix[matrix>253] = 255
        matrix[(matrix>=2) & (matrix<=253)] = 125
        # print(np.unique(matrix))
        # plt.figure()
        # plt.imshow(matrix, cmap="gray")
        # plt.show()
        self.matrix = matrix

    # 生成随机种群
    def init_random_solution(self):
        good = np.where(self.matrix==255)
        bad = np.where(self.matrix==0)
        self.good = list(zip(good[0],good[1]))
        self.bad = list(zip(bad[0],bad[1]))
        for i in range(self.number):
            solution = []
            if len(self.good) >= len(self.bad):
                temp = copy.deepcopy(self.good)
                for b in self.bad:
                    g = random.sample(temp,1)[0]
                    temp.remove(g)
                    solution.append([g,b])
            else:
                temp = copy.deepcopy(self.bad)
                for g in self.good:
                    b = random.sample(temp,1)[0]
                    temp.remove(b)
                    solution.append([g,b])
            self.solutions.append(solution)
            self.remains.append(temp)
            # print(solution)

    # 适应度函数
    def total_distance(self, solution):
        distance = 0
        for s in solution:
            distance += ((s[0][0]-s[1][0])**2+(s[0][1]-s[1][1])**2)**0.5
        return 1/distance

    def show_result(self, solution):
        plt.figure()
        plt.imshow(self.matrix, cmap="gray")
        for s in solution:
            y =[s[0][0],s[1][0]]
            x =[s[0][1],s[1][1]]
            plt.plot(x,y,'r')
        # plt.show()
        if not os.path.exists('log'): os.makedirs('log')
        plt.savefig('./log/{}.png'.format(self.iter))

    # 判断是否存在一连多的情况
    def judge_solution(self, solution):
        good = [s[0] for s in solution]
        bad = [s[1] for s in solution]
        good_temp = []
        for g in good:
            if g not in good_temp:
                good_temp.append(g)
        bad_temp = []
        for b in bad:
            if b not in bad_temp:
                bad_temp.append(b)
        if len(good) != len(good_temp) or len(bad) != len(bad_temp):
            return False
        else:
            return True

    # 遗传进化
    def evolution(self):
        distances = [self.total_distance(solution) for solution in self.solutions]
        self.solutions = [x for _,x,_ in sorted(zip(distances,self.solutions,self.remains),reverse=True)]
        self.remains = [x for _,_,x in sorted(zip(distances,self.solutions,self.remains),reverse=True)]
        distances = [x for x,_,_ in sorted(zip(distances,self.solutions, self.remains),reverse=True)]
        print(self.iter, 1/max(distances), (self.rounds-self.iter)/self.rounds)

        self.show_result(self.solutions[0])

        probability = [d/sum(distances) for d in distances]
        probability = [np.sum(probability[:i+1]) for i in range(len(probability))]
        next = copy.deepcopy(self.solutions[:int(self.number*0.3)])

        while len(next) < self.number:
            remain = []

            p = np.random.uniform()
            i = np.where(np.array(probability)>p)[0][0]
            f = self.solutions[i]
            remain = self.remains[i]

            p = np.random.uniform()
            i = np.where(np.array(probability)>p)[0][0]
            m = self.solutions[i]
            remain.extend([r for r in self.remains[i] if r not in remain])

            cross_index = np.random.randint(2,size=(len(self.solutions[0])))

            c1 = copy.deepcopy(f)

            c2 = copy.deepcopy(m)

            if len(self.good) >= len(self.bad):
                c1_g = [c[0] for c in c1]
                c1_t = [c1_g[i] for i in range(len(cross_index)) if cross_index[i]== 1]

                c2_g = [c[0] for c in c2]
                c2_t = [c2_g[i] for i in range(len(cross_index)) if cross_index[i]== 1]

                for t in c1_t:
                    i = c1_g.index(t)
                    if t in c2_g:
                        j = c2_g.index(t)
                        temp = c1[i]
                        c1[i] = c1[j]
                        c1[j] = temp

                for t in c2_t:
                    i = c2_g.index(t)
                    if t in c1_g:
                        j = c1_g.index(t)
                        temp = c2[i]
                        c2[i] = c2[j]
                        c2[j] = temp
            else:
                c1_b = [c[1] for c in c1]
                c1_t = [c1_b[i] for i in range(len(cross_index)) if cross_index[i]== 1]

                c2_b = [c[1] for c in c2]
                c2_t = [c2_b[i] for i in range(len(cross_index)) if cross_index[i]== 1]

                for t in c1_t:
                    i = c1_b.index(t)
                    if t in c2_b:
                        j = c2_b.index(t)
                        temp = c1[i]
                        c1[i] = c1[j]
                        c1[j] = temp

                for t in c2_t:
                    i = c2_b.index(t)
                    if t in c1_b:
                        j = c1_b.index(t)
                        temp = c2[i]
                        c2[i] = c2[j]
                        c2[j] = temp

            # next.extend([c1,c2])
            if len(self.good) != len(self.bad):
                if np.random.uniform() > (self.rounds-self.iter)/self.rounds:
                    index = np.random.randint(len(self.solutions[0]),size=1)[0]
                    if len(self.good) > len(self.bad):
                        c1[index][0] = random.sample(remain, 1)[0]
                    else:
                        c1[index][1] = random.sample(remain, 1)[0]
                if np.random.uniform() > (self.rounds-self.iter)/self.rounds:
                    index = np.random.randint(len(self.solutions[0]),size=1)[0]
                    if len(self.good) > len(self.bad):
                        c2[index][0] = random.sample(remain, 1)[0]
                    else:
                        c2[index][1] = random.sample(remain, 1)[0]
            if self.judge_solution(c1):
                if (self.total_distance(c1) > self.total_distance(f)) or (self.total_distance(c1) > self.total_distance(m)):
                    next.append(c1)
            if self.judge_solution(c2):
                if (self.total_distance(c2) > self.total_distance(f)) or (self.total_distance(c2) > self.total_distance(m)):
                    next.append(c2)
        self.solutions = next
        self.iter += 1

    def main(self):
        self.init_random_matrix()
        self.init_random_solution()
        for i in range(self.rounds):
            self.evolution()


if __name__ == "__main__":
    # number, rounds
    GA = GeneticAlgorithm(2000,1000)
    GA.main()
