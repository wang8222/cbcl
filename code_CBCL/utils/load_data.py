import numpy as np
import scipy.sparse as sp
import torch as th
from sklearn.preprocessing import OneHotEncoder
 
def encode_onehot(labels):
    labels = labels.reshape(-1, 1)  # 将标签数组重塑为一列，每行一个标签
    enc = OneHotEncoder()  # 创建一个OneHotEncoder对象
    enc.fit(labels)  # 使用标签数据来训练编码器
    labels_onehot = enc.transform(labels).toarray()  # 将标签转换为one-hot编码，并转换为数组形式
    return labels_onehot  # 返回one-hot编码的标签


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))  # 计算特征矩阵每行的和
    r_inv = np.power(rowsum, -1).flatten()  # 计算每行和的倒数，并将结果展平为一维数组
    r_inv[np.isinf(r_inv)] = 0.  # 如果有无穷大的值，将其替换为0
    r_mat_inv = sp.diags(r_inv)  # 创建一个对角矩阵，对角线上的元素是r_inv
    features = r_mat_inv.dot(features)  # 用r_mat_inv左乘特征矩阵，实现行归一化
    return features.todense()  # 返回密集矩阵形式的特征
#是对特征矩阵进行行归一化处理

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)  # 将scipy稀疏矩阵转换为COO格式，并将数据类型转换为float32
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))  # 将行索引和列索引堆叠在一起，并转换为PyTorch张量
    values = th.from_numpy(sparse_mx.data)  # 将数据转换为PyTorch张量
    shape = th.Size(sparse_mx.shape)  # 获取稀疏矩阵的形状，并转换为PyTorch的Size对象
    return th.sparse.FloatTensor(indices, values, shape)  # 创建一个PyTorch稀疏张量
#将scipy的稀疏矩阵转换为PyTorch的稀疏张量


def load_acm(ratio, type_num):
    path = "../data/acm/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    # 加载邻接矩阵和特征矩阵
    nei_a  = sp.load_npz(path + "nei_a.npz")
    nei_s = sp.load_npz(path + "nei_s.npz")
    feat_p = sp.load_npz(path + "p_feat.npz")
    feat_a = sp.load_npz(path + "a_feat.npz")
    feat_s = sp.eye(type_num[2])# 创建一个单位矩阵，大小为type_num[2] x type_num[2]
    pap = sp.load_npz(path + "pap.npz")
    psp = sp.load_npz(path + "psp.npz")
    # 加载训练集、测试集和验证集
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]
    # 将数据转换为PyTorch张量
    label = th.FloatTensor(label)
    nei_a = sparse_mx_to_torch_sparse_tensor(nei_a)
    nei_s = sparse_mx_to_torch_sparse_tensor(nei_s)
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_s = th.FloatTensor(preprocess_features(feat_s))

    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]

    return [nei_a, nei_s], [feat_p, feat_a, feat_s], [pap, psp], label, train, val, test

#ratio是一个列表，表示训练集、测试集和验证集的比例
def load_dblp(ratio, type_num):
    path = "../data/dblp/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    feat_a = sp.load_npz(path + "a_feat.npz").astype("float32")
    feat_p = sp.load_npz(path + "p_feat.npz").astype("float32")
    feat_c = sp.eye(type_num[3])
    feat_t = np.load(path+"t_feat.npz")

    nei_ap = sp.load_npz(path + "nei_ap.npz")
    nei_apc = sp.load_npz(path + "nei_apc.npz")
    nei_apcp = sp.load_npz(path + "nei_apcp.npz")
    nei_apt = sp.load_npz(path + "nei_apt.npz")
    nei_aptp = sp.load_npz(path + "nei_aptp.npz")

    apa = sp.load_npz(path + "apa.npz")  
    apcpa = sp.load_npz(path + "apcpa.npz")
    aptpa = sp.load_npz(path + "aptpa.npz") 

    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]
    
    label = th.FloatTensor(label)

    nei_ap = sparse_mx_to_torch_sparse_tensor(nei_ap)
    nei_apc = sparse_mx_to_torch_sparse_tensor(nei_apc)
    nei_apcp = sparse_mx_to_torch_sparse_tensor(nei_apcp)
    nei_apt = sparse_mx_to_torch_sparse_tensor(nei_apt)
    nei_aptp = sparse_mx_to_torch_sparse_tensor(nei_aptp)
        
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_t = th.FloatTensor(feat_t)
    feat_c = th.FloatTensor(preprocess_features(feat_c))

    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]

    return [nei_ap, nei_apc, nei_apcp, nei_apt, nei_aptp], [feat_a, feat_p, feat_t, feat_c], [apa, apcpa, aptpa], label, train, val, test


def load_aminer(ratio, type_num):
    path = "../data/aminer/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    nei_a = sp.load_npz(path + "nei_a.npz")
    nei_r = sp.load_npz(path+ "nei_r.npz")
 
    feat_p_pap = np.load(path + "feat_p_pap.w1000.l100.npy").astype('float')
    feat_p_prp = np.load(path + "feat_p_prp.w1000.l100.npy").astype('float')
    feat_a = np.load(path + "feat_a.w1000.l100.npy").astype('float')
    feat_r = np.load(path + "feat_r.w1000.l100.npy").astype('float')

    feat_p = th.stack((th.FloatTensor(feat_p_pap),th.FloatTensor(feat_p_prp)))
    feat_a = th.FloatTensor(feat_a)
    feat_r = th.FloatTensor(feat_r)

    pap = sp.load_npz(path + "pap.npz")
    prp = sp.load_npz(path + "prp.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    nei_a = sparse_mx_to_torch_sparse_tensor(nei_a)
    nei_r = sparse_mx_to_torch_sparse_tensor(nei_r)

    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    return [nei_a, nei_r], [feat_p, feat_a, feat_r], [pap, prp], label, train, val, test

def load_data(dataset, ratio, type_num):
    if dataset == "acm":
        data = load_acm(ratio, type_num)
    elif dataset == "dblp":
        data = load_dblp(ratio, type_num)
    elif dataset == "aminer":
        data = load_aminer(ratio, type_num)
    return data
