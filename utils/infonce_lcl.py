import torch
import torch.nn as nn
import torch.nn.functional as F 


class InfoNCE_LCL(nn.Module):
    def __init__(self, tau=0.05):
        super().__init__()
        self.tau = tau  # 温度系数
    
    def normalize(self, *xs):
        return [None if x is None else F.normalize(x, dim=-1) for x in xs]
    
    def sim_loss(self, query, pos_key, neg_keys):
        '''
        query: [batch, dim]
        pos_key: [batch, dim]
        neg_key: [batch, neg_num, dim]
        '''
        f = lambda x: torch.exp(x / self.tau)
        query, pos_key, neg_keys = self.normalize(query, pos_key, neg_keys)
        pos_sim = f(torch.sum(query * pos_key, dim=1)) # [batch, 1]
        query = query.unsqueeze(1) # [batch, 1, dim]
        neg_keys = neg_keys.transpose(-2, -1) # [batch, dim, neg_num]
        neg_sim = f(query @ neg_keys) # [batch, 1, neg_num]
        neg_sim = neg_sim.squeeze(1) # [batch, neg_num]
        loss_all = -torch.log(pos_sim / torch.sum(neg_sim, dim=1))
        return torch.sum(loss_all) / len(loss_all)

    def infonce_lcl_loss(self, query, pos_key, neg_keys):
        neg_keys_list = []
        for i in range(len(neg_keys)):
            temp = neg_keys[i]
            for i in temp: neg_keys_list.append(i)
        pos_key = torch.stack(pos_key)
        neg_keys = torch.stack(neg_keys_list)
        query, pos_key, neg_keys = self.normalize(query, pos_key, neg_keys)
        pos_logit = torch.sum(query * pos_key, dim=1, keepdim=True) # [batch, 1]
        query = query.unsqueeze(1)
        neg_keys = neg_keys.transpose(-2, -1)
        neg_logits = query @ neg_keys
        neg_logits = neg_logits.squeeze(1) # [batch, neg_num]
        logits = torch.cat([pos_logit, neg_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
        return F.cross_entropy(logits / self.tau, labels, reduction='mean')


def nearest_client():
    # 创建一个随机的列表，包含20个维度为[10,]的张量
    num_elements = 20
    embedding_dim = 10
    a = torch.randn(num_elements, embedding_dim)
    # 计算两两元素之间的欧几里得距离
    # 要获取欧几里得距离，我们可以计算L2范数的平方和（L2距离的平方）然后取平方根
    distances = torch.cdist(a, a, p=2)  # cdist用于计算两个矩阵之间的距离，参数p=2表示L2范数
    # 对角线是元素自身与自身的距离，设置为无穷大，避免找最小距离时找到自身
    distances.fill_diagonal_(float("inf"))
    # 找到每个元素最相似的那个元素
    min_distances, min_indices = torch.min(distances, dim=1)
    print("最相似元素的距离:", min_distances)
    print("最相似元素的索引:", min_indices)


if __name__=='__main__':
    # loss_func = InfoNCE_LCL(tau=0.08)
    # query = torch.randn([8,10])
    # pos_key = torch.randn([8,10])
    # neg_keys = torch.randn([8,8,10])
    # loss_lcl = loss_func.sim_loss(query, pos_key, neg_keys)
    # print(loss_lcl)
    # loss_infonce = loss_func.infonce_loss(query, pos_key, neg_keys)
    # print(loss_infonce)
    nearest_client()