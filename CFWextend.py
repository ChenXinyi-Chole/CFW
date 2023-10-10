import itertools
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import math
import random
import scipy.io
from scipy.optimize import line_search
from Feature_Selection.tool import normalization
from scipy.sparse import *
from sklearn.metrics.pairwise import pairwise_distances
from Feature_Selection.tool import generate_constrains


def extend_MC1(X, M, k):
    '''
    对每一个cannot-link(x,y), 分别找 x和 y的 k近邻集合A、B
    M' = M ∪ {(x,a),(y,b)|a∈A/x, b∈B/y}
    C' = C ∪ {(a,b)| a∈A/x, b∈B/y}
    '''
    # 作用：使用欧几里得距离，返回X中距离Y最近点的索引，shape与X一致
    # 过程：逐个查找X列表中的点，返回距离Y列表每个点最近的X点索引。
    # d = pairwise_distances_argmin(X, X,metric='euclidean')
    M1, C1 = [],[]
    for m in M:
        M1.append(list(m))

    D = pairwise_distances(X, metric="euclidean")
    idx = np.argsort(D, axis=1)
    # print('idx,', idx)
    idx_new = idx[:, 1: k + 1]
    # print('idx_new,', idx_new)
    for m in M:
        m = list(m)
        # print(c)
        A = list(idx_new[m[0]-1])
        B = list(idx_new[m[1]-1])
        for i in range(k):
            M1.append([m[0], A[i]+1])
            M1.append([m[1], B[i]+1])
        AB = [A, B]
        # print(AB)
        cc = itertools.product(*AB)
        for iter in cc:
            # print(iter)
            C1.append([iter[0]+1,iter[1]+1])
        # print("M1:",M1)
        # print("C1:", C1)
    return M1


def construct_LM_matrix(X, M):
    n_samples, n_features = np.shape(X)
    S = np.zeros((n_samples, n_samples))

    for item in M:
        S[item[0]-1][item[1]-1] = 1
        S[item[1]-1][item[0]-1] = 1
    E = np.ones(n_samples)
    D = diags(S@E)
    L_m = D-S
    return L_m


def nearhitSet(X, num, w):
    '''
    :param X: dataset
    :param num: the number of a constraint
    :return:
    '''
    # nearhit: the distance to its nearest sample of same class label用最近邻样本表示
    # 将样本映射到特征空间下
    # print("X",X)
    # print("w",w)
    X_new = np.multiply(X, w)
    # print("X2", X)
    D = pairwise_distances(X_new)
    # D **= 2
    # sort the distance matrix D in ascending order
    # dump = np.sort(D, axis=1)
    idx = np.argsort(D, axis=1)
    # 为每个constraint选10个近邻
    nearhitset = idx[num][1:6]
    return nearhitset


def hit_disctace(X, num, nearhitset, w, sigma):
    # X = np.multiply(X, w)
    weight = [] # 描述每个样本的概率，个数==样本个数
    distances = []
    for nearhit in nearhitset:
        # print("np. multiply(w, list(X[num] - X[nearhit]))",np. multiply(w, list(X[num] - X[nearhit])))
        abs_dis = [abs(i) for i in list(X[num] - X[nearhit])]
        w_i = np.exp(-sum(np. multiply(w, abs_dis))/sigma)
        weight.append(w_i)
        # dis = [abs(i) for i in list(X[num] - X[nearhit])]  # 是一个向量
        distances.append(abs_dis)
    # print("sum(weight)",sum(weight))
    sum_weight_before = sum(weight)
    for i in range(len(weight)):
        # print("weight[i]",weight[i])
        if sum_weight_before != 0:
            weight[i] = weight[i]/sum_weight_before
        else:
            weight[i] = w[i]
    # print(weight)
    # print(np.array(distances))
    # print(np.array(distances).shape)
    hit_dis = [0] * len(w)

    for cnt in range(len(weight)):  # 有多少个近邻样本，每个近邻样本有其权重
        # print(distances[cnt])
        # print(weight[cnt])
        distances[cnt] = [i * weight[cnt] for i in distances[cnt]]
        hit_dis = np.add(hit_dis, distances[cnt])
        # print(hit_dis) i/5 for i in hit_dis]
    return hit_dis


def updatew(w, z,  lambda1, lambda2, MM):
    # print("w",w)
    # print("z",z)
    # print("-np. dot(w,z)",-np. dot(w,z))
    # print()
    res1 = np.multiply(z, np.exp(-np. dot(w,z)/(1+np.exp(-np. dot(w,z)))))
    # print("res1",res1)
    res2 = lambda1
    # print("w",w)
    # print("MM",MM)
    # print("np.multiply(w,MM)",np.multiply(w,MM))
    # print("res1",res1)
    res3 = 2*lambda2*np.multiply(w,MM)
    # print("res2",res2)
    # print("res3",res3)
    # print("res1+res2+res3",res1+res2+res3)
    return res2+res3-res1


def CFW(X, M, C, sigma, theta_, T, lambda1, lambda2, learningrate):
    t, theta = 1, 1+theta_
    L_m = construct_LM_matrix(X, M)
    M = np.transpose(X) @ L_m @ X
    MM = np.diag(M)

    n_samples, n_features = np.shape(X)
    w = [1] * n_features  # w代表每个特征的权重

    while t <= T and theta > theta_:
        # print("z", z)
        z = [0] * n_features  # z存储权重的累积过程
        for c in C:
            x, y = c[0]-1, c[1]-1
            Hsetx = nearhitSet(X, x, w)  # nearhit of x 权重空间下的近邻
            Hsety = nearhitSet(X, y, w)  # nearmiss of x 权重空间下的近邻
            dis_x2Hy = hit_disctace(X, x, Hsety, w, sigma)
            # print("dis_x2Hy", dis_x2Hy)
            dis_x2Hx = hit_disctace(X, x, Hsetx, w, sigma)
            # print("dis_x2Hx",dis_x2Hx)
            diff_dis = np.add(np.array(dis_x2Hy), -np.array(dis_x2Hx), dtype=np.float64)
            z = np.add(z, diff_dis, dtype=np.float64)
        w0 = w

        w = w0 - learningrate * updatew(w0, z, lambda1, lambda2, MM)

        for i in range(n_features):
            if w[i] < 0:
                w[i] = 0
        # print("w-w_t",w-w_t)

        theta = np.linalg.norm(np.array(w)-np.array(w0))
        t += 1

    id = np.argsort(w)
    idx = id[::-1]
    return np.sort(w), idx


