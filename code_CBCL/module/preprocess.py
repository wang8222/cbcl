# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp
import random
import torch

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):  # 如果输入的稀疏矩阵不是COO格式
        sparse_mx = sparse_mx.tocoo()  # 将其转换为COO格式
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()  # 将行索引和列索引堆叠在一起，并转置，得到坐标
    values = sparse_mx.data  # 获取数据
    shape = sparse_mx.shape  # 获取形状
    return coords, values, shape  # 返回坐标、数据和形状
#将稀疏矩阵转换为元组形式

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)  # 将邻接矩阵转换为COO格式
    rowsum = adj.sum(axis=1)  # 计算邻接矩阵每行的和
    rowsum = np.array(adj.sum(1))  # 将每行的和转换为数组
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # 计算每行和的平方根的倒数，并将结果展平为一维数组
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.  # 如果有无穷大的值，将其替换为0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # 创建一个对角矩阵，对角线上的元素是d_inv_sqrt
    adj_norm_coo = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo().todense()  # 对邻接矩阵进行归一化处理，并将结果转换为密集矩阵形式

    adj_torch = torch.from_numpy(adj_norm_coo).float()  # 将归一化后的邻接矩阵转换为PyTorch张量，并将数据类型转换为float
    if torch.cuda.is_available():  # 如果有可用的GPU
        adj_torch = adj_torch.cuda()  # 将张量移动到GPU上
    return adj_torch  # 返回归一化后的邻接矩阵
#对邻接矩阵进行对称归一化处理

def mask_edges(adjs, sub_num, adj_mask):
    mask_adjs=[]  # 初始化一个空列表，用于存储掩码后的邻接矩阵
    for i in range(sub_num):  # 对每个子图进行遍历
        adj = adjs[i]  # 获取当前子图的邻接矩阵
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)  # 去除邻接矩阵的对角线元素
        adj.eliminate_zeros()  # 去除邻接矩阵中的零元素
        adj_tuple = sparse_to_tuple(adj)  # 将邻接矩阵转换为元组形式
        edges = adj_tuple[0]  # 获取边的坐标
        np.random.shuffle(edges)  # 随机打乱边的顺序
        mask_edges_num = int(adj_mask[i]*len(edges))  # 计算需要掩码的边的数量
        rest_edges = edges[mask_edges_num:]  # 获取剩余的边
        data = np.ones(rest_edges.shape[0])  # 创建一个全为1的数组，大小等于剩余边的数量
        adj = sp.coo_matrix((data, (rest_edges[:, 0], rest_edges[:, 1])), shape=adjs[i].shape)  # 创建一个新的邻接矩阵，只包含剩余的边
        adj = (adj + np.eye(adjs[i].shape[0]))  # 在新的邻接矩阵的对角线上添加1
        adj = normalize_adj(adj)  # 对新的邻接矩阵进行归一化处理
        mask_adjs.append(adj)  # 将新的邻接矩阵添加到列表中
    return mask_adjs  # 返回掩码后的邻接矩阵列表
#对邻接矩阵进行掩码处理

def mask_feature(feat, adj_mask):
    feats_coo = sp.coo_matrix(feat)  # 将特征矩阵转换为COO格式
    feats_num = feats_coo.getnnz()  # 获取特征矩阵中非零元素的数量
    feats_idx = [i for i in range(feats_num)]  # 创建一个列表，包含从0到feats_num-1的所有整数
    mask_num = int(feats_num * adj_mask)  # 计算需要掩码的特征的数量
    mask_idx = random.sample(feats_idx, mask_num)  # 从feats_idx中随机抽取mask_num个元素，得到需要掩码的特征的索引
    feats_data = feats_coo.data  # 获取特征矩阵的数据
    for j in mask_idx:  # 对每个需要掩码的特征进行遍历
        feats_data[j] = 0  # 将该特征的值设置为0
    mask_feats = torch.sparse.FloatTensor(torch.LongTensor([feats_coo.row.tolist(), feats_coo.col.tolist()]),
                          torch.FloatTensor(feats_data.astype(np.float64)))  # 创建一个PyTorch稀疏张量，表示掩码后的特征矩阵

    if torch.cuda.is_available():  # 如果有可用的GPU
        mask_feats = mask_feats.cuda()  # 将张量移动到GPU上
    return mask_feats  # 返回掩码后的特征矩阵
#对特征矩阵进行掩码处理

def mask_features(feats, adj_mask):
    if len(feats.size()) == 3:
        mask_feats = []
        for feat in feats:
            mask_feats.append(mask_feature(feat,adj_mask))
        if torch.cuda.is_available():
            mask_feats =  [f.cuda() for f in mask_feats]
    else:
        mask_feats = mask_feature(feats, adj_mask)
        if torch.cuda.is_available():
            mask_feats = mask_feats.cuda() 
    return mask_feats
#对特征矩阵进行掩码处理，三维还是二维

def pathsim(adjs, max_nei):
    print("the number of edges:", [adj.getnnz() for adj in adjs])  # 打印每个邻接矩阵的边的数量
    top_adjs = []  # 初始化一个空列表，用于存储最大邻居数的邻接矩阵
    adjs_num = []  # 初始化一个空列表，用于存储每个邻接矩阵的非零元素的数量
    for t in range(len(adjs)):  # 对每个邻接矩阵进行遍历
        A = adjs[t].todense()  # 将当前邻接矩阵转换为密集矩阵形式
        value = []  # 初始化一个空列表，用于存储路径相似度的值
        x,y = A.nonzero()  # 获取邻接矩阵中非零元素的坐标
        for i,j in zip(x,y):  # 对每个非零元素进行遍历
            value.append(2 * A[i, j] / (A[i, i] + A[j, j]))  # 计算路径相似度，并将结果添加到列表中
        pathsim_matrix = sp.coo_matrix((value, (x, y)), shape=A.shape).toarray()  # 创建一个新的稀疏矩阵，表示路径相似度矩阵，并将其转换为数组形式
        idx_x = np.array([np.ones(max_nei[t])*i for i in range(A.shape[0])], dtype=np.int32).flatten()  # 创建一个数组，表示每个节点的最大邻居数
        idx_y = np.sort(np.argsort(pathsim_matrix, axis=1)[:,::-1][:,0:max_nei[t]]).flatten()  # 获取路径相似度矩阵中每行最大的max_nei[t]个元素的索引，并将结果展平为一维数组
        new = []  # 初始化一个空列表，用于存储新的邻接矩阵的数据
        for i,j in zip(idx_x,idx_y):  # 对每个索引进行遍历
            new.append(A[i,j])  # 获取当前索引对应的元素，并将其添加到列表中
        new = (np.int32(np.array(new)))  # 将列表转换为数组，并将数据类型转换为int32
        adj_new = sp.coo_matrix((new, (idx_x,idx_y)), shape=adjs[t].shape)  # 创建一个新的稀疏矩阵，表示新的邻接矩阵
        adj_num = np.array(new).nonzero()  # 获取新的邻接矩阵中非零元素的数量
        adjs_num.append(adj_num[0].shape[0])  # 将非零元素的数量添加到列表中
        top_adjs.append(adj_new)  # 将新的邻接矩阵添加到列表中
    print("the top-k number of edges:", [adj for adj in adjs_num])  # 打印每个邻接矩阵的最大邻居数的边的数量
    return top_adjs  # 返回最大邻居数的邻接矩阵列表
#计算邻接矩阵的路径相似度，并返回最大邻居数的邻接矩阵