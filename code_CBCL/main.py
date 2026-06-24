import numpy
import torch
from utils import load_data, set_params, evaluate, run_kmeans
from module.meow import MEOW
from module.preprocess import *
import warnings
import datetime
import pickle as pkl
import random

warnings.filterwarnings('ignore')
args = set_params()
if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")

## random seed ##
seed = args.seed
numpy.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def train():
    # 加载数据
    nei_index, feats, adjs, label, idx_train, idx_val, idx_test = \
        load_data(args.dataset, args.ratio, args.type_num)
    nb_classes = label.shape[-1]  # 类别数量
    if args.dataset == 'aminer':  # 如果数据集是aminer
        feats_dim_list = [64, 64, 64]  # 特征维度列表
    else:
        feats_dim_list = [i.shape[1] for i in feats]  # 特征维度列表
    sub_num = int(len(adjs))  # 子图数量
    print("Dataset: ", args.dataset)  # 打印数据集名称
    print("The number of meta-paths: ", sub_num)  # 打印元路径数量
    print("The dim of different kinds' nodes' feature: ", feats_dim_list)  # 打印不同类型节点的特征维度
    feat = feats[0]  # 获取第一个特征
    adjs = pathsim(adjs, args.nei_max)  # 计算邻接矩阵的路径相似度
    mask_feat = mask_features(feat, args.feat_mask)  # 对特征进行掩码处理
    adjs_norm = [normalize_adj(adj) for adj in adjs]  # 对邻接矩阵进行归一化处理
    mask_adjs = mask_edges(adjs, sub_num, args.adj_mask)  # 对邻接矩阵进行掩码处理
    print("Feature and Edge Mask Finished!")  # 打印特征和边掩码完成的信息

    # 创建模型
    model = MEOW(feats_dim_list, sub_num, args.hidden_dim, args.embed_dim, args.tau, adjs_norm, args.lam_proto, \
                 args.dropout, nei_index, args.dataset)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)  # 创建优化器

    if torch.cuda.is_available():  # 如果有可用的GPU
        print('Using CUDA')  # 打印使用CUDA的信息
        model.cuda()  # 将模型移动到GPU上
        feat = feat.cuda()  # 将特征移动到GPU上
        feats = [f.cuda() for f in feats]  # 将所有的特征移动到GPU上
        label = label.cuda()  # 将标签移动到GPU上
        idx_train = [i.cuda() for i in idx_train]  # 将训练集的索引移动到GPU上
        idx_val = [i.cuda() for i in idx_val]  # 将验证集的索引移动到GPU上
        idx_test = [i.cuda() for i in idx_test]  # 将测试集的索引移动到GPU上

    cnt_wait = 0  # 初始化等待计数器
    best = 1e9  # 初始化最佳损失值

    starttime = datetime.datetime.now()  # 获取开始时间
    epoch_times = args.nb_epochs  # 获取训练轮数

    num_clusters = args.num_cluster  # 获取聚类数量
    for epoch in range(epoch_times):  # 对每个训练轮进行遍历
        if not args.save_emb:  # 如果不保存嵌入
            break  # 跳出循环
        print("---------------------------------------------------")  # 打印分隔线
        print("Epoch:", epoch)  # 打印当前训练轮数
        model.train()  # 将模型设置为训练模式
        optimizer.zero_grad()  # 清零优化器的梯度
        loss = model(feats, mask_feat, mask_adjs, adjs_norm, num_clusters)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        print('best:', best)  # 打印最佳损失值
        if best > loss:  # 如果当前损失值小于最佳损失值
            best = loss  # 更新最佳损失值
            cnt_wait = 0  # 重置等待计数器
        else:
            cnt_wait += 1  # 等待计数器加1
        # print('current patience: ', cnt_wait)
        if cnt_wait >= args.patience:  # 如果等待计数器大于等于容忍度
            print('Early stopping!')  # 打印提前停止的信息
            break  # 跳出循环

    if args.save_emb:  # 如果保存嵌入
        print("Start to save embeds.")  # 打印开始保存嵌入的信息
        embeds = model.get_embeds()  # 获取嵌入
        f = open("./embeds/" + args.dataset + "/" + str(args.turn) + ".pkl", "wb")  # 打开文件
        pkl.dump(embeds.cpu().data.numpy(), f)  # 将嵌入保存到文件中
        f.close()  # 关闭文件
        print("Save finish.")  # 打印保存完成的信息
        run_kmeans(embeds.cpu(), torch.argmax(label.cpu(), dim=-1), nb_classes, starttime, args.dataset)  # 运行K-means聚类
    else:
        print("Read embeds.")  # 打印读取嵌入的信息
        file = open("./embeds/" + args.dataset + "/" + str(args.turn) + ".pkl", "rb")  # 打开文件
        embeds = torch.from_numpy(pkl.load(file)).cuda()  # 从文件中读取嵌入，并将其转换为PyTorch张量
        file.close()  # 关闭文件

    for i in range(len(idx_train)):  # 对每个训练集进行遍历
        evaluate(embeds, args.ratio[i], idx_train[i], idx_val[i], idx_test[i], label, nb_classes, device, args.dataset,
                 args.eva_lr, args.eva_wd, starttime)  # 进行评估
    endtime = datetime.datetime.now()  # 获取结束时间
    time = (endtime - starttime).seconds  # 计算总时间
    print("Total time: ", time, "s")  # 打印总时间


if __name__ == '__main__':
    train()