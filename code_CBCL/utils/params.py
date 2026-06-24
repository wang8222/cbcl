import argparse
import sys

argv = sys.argv
dataset = "acm"

def acm_params():
    parser = argparse.ArgumentParser() # 创建一个命令行参数解析器
    parser.add_argument('--save_emb', action="store_false")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="acm")
    parser.add_argument('--ratio', type=int, default=[20,40,60]) # 训练集、验证集和测试集的比例，默认为[20, 40, 60]
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nb_epochs', type=int, default=10000)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=64)
    
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.03)# 评估的学习率，默认为0.03
    parser.add_argument('--eva_wd', type=float, default=0)# 评估的权重衰减，默认为0
    
    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=40) # 容忍度，默认为20  10-40 步长5
    parser.add_argument('--l2_coef', type=float, default=0)# L2正则化系数，默认为0 0.0004 0.003
    parser.add_argument('--lr', type=float, default=0.0007) # 学习率，默认为0.0007  0.0005 0.0006
    parser.add_argument('--dropout', type=float, default=0.2)# Dropout率，默认为0.2  0-0.9
 
    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.4)
    parser.add_argument('--feat_mask', type=float, default=0.3) # 特征掩码，默认为0.3   0.1-0.8
    parser.add_argument('--adj_mask', type=float, default=[0.3,0.2]) # 邻接矩阵掩码，默认为[0.3,0.2]
    parser.add_argument('--nei_max', type=int, default=[110,700]) # 最大邻居数，默认为[110,700]
    parser.add_argument('--num_cluster', default=[100,300], type=int, help='number of clusters')   # 聚类的数量，默认为[100,300]
    parser.add_argument('--lam_proto', type=float, default=1)    #0.1 1 10
    
    args, _ = parser.parse_known_args()
    args.type_num = [4019, 7167, 60]  # the number of every node type 每种节点类型的数量
    args.nei_num = 2  # the number of neighbors' types邻居类型的数量
    return args


def dblp_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_false")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="dblp")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--nb_epochs', type=int, default=10000)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=64)
    
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)
    
    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=35)
    parser.add_argument('--l2_coef', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.0006)
    parser.add_argument('--dropout', type=float, default=0.2)
    
    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.9)
    parser.add_argument('--feat_mask', type=float, default=0.2)
    parser.add_argument('--adj_mask', type=float, default=[0.2,0.5,0.6])
    parser.add_argument('--lam_proto', type=float, default=1)
    parser.add_argument('--nei_max', type=int, default=[25,200,40])
    parser.add_argument('--num_cluster', default=[200,700])

    args, _ = parser.parse_known_args()
    args.type_num = [4057, 14328, 7723, 20]  # the number of every node type
    args.nei_num = 1  # the number of neighbors' types
    return args

def aminer_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_false")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="aminer")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=10000)
    
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.1)
    parser.add_argument('--eva_wd', type=float, default=8e-4)
    
   # The parameters of learning process
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--l2_coef', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.0007)
    parser.add_argument('--dropout', type=float, default=0.2)
    
    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.9)
    parser.add_argument('--feat_mask', type=float, default=0.2)
    parser.add_argument('--adj_mask', type=float, default=[0.7,0.4])
    parser.add_argument('--nei_max', type=int, default=[5,21])
    parser.add_argument('--num_cluster', default=[500,1200], type=int, help='number of clusters')
    parser.add_argument('--lam_proto', type=float, default=0.1)
     
    args, _ = parser.parse_known_args()
    args.type_num = [6564, 13329, 35890]  # the number of every node type
    args.nei_num = 2  # the number of neighbors' types
    return args

def set_params():
    if dataset == "acm":
        args = acm_params()
    elif dataset == "dblp":
        args = dblp_params()
    elif dataset == "aminer":
        args = aminer_params()
    return args
