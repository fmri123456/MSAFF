import torch
import torch.nn as nn
import torch.nn.functional as F
from CNN import CNN
from MultiHeadAttention import Multiheads_Attention
from SelfAttention import Self_Attention


class CrossAttention(nn.Module):
    def __init__(self, input_dim, pool_size=2, pool_stride=2):
        super(CrossAttention, self).__init__()
        self.input_dim = input_dim
        self.pool_size = pool_size
        self.pool_stride = pool_stride

        # 线性层用于将输入转换为 Q, K, V 向量
        self.linear_q = nn.Linear(input_dim, input_dim)
        self.linear_k = nn.Linear(input_dim, input_dim)
        self.linear_v = nn.Linear(input_dim, input_dim)
        # 池化层，用于池化Q,K,V向量
        self.maxpool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)

    def forward(self, input):
        # 计算 Q, K, V 向量
        Q = self.linear_q(input)
        K = self.linear_k(input)
        V = self.linear_v(input)
        # 对Q,K向量进行池化
        Q_pooled = self.maxpool(Q.permute(0, 2, 1)).permute(0, 2, 1)
        K_pooled = self.maxpool(K.permute(0, 2, 1)).permute(0, 2, 1)
        V_pooled = self.maxpool(V.permute(0, 2, 1)).permute(0, 2, 1)
        return Q_pooled, K_pooled, V_pooled

class CrossModalFusion(nn.Module):
    def __init__(self, input_dim):
        super(CrossModalFusion, self).__init__()
        self.input_dim = input_dim

    def forward(self, Q1, K1, V1, Q2, K2, V2):
        # 交叉拼接
        fusion_Q = torch.cat((Q1, Q2), dim=1)
        fusion_K = torch.cat((K1, K2), dim=1)
        fusion_V = torch.cat((V1, V2), dim=1)

        # 计算注意力权重
        raw_weights = torch.matmul(fusion_Q, fusion_K.permute(0, 2, 1)) / (self.input_dim ** 0.5)
        attn_weights = F.softmax(raw_weights, dim=-1)
        # 加权求和
        attn_outputs = torch.matmul(attn_weights, fusion_V)
        return attn_outputs
class MWSAFM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=2):
        super(MWSAFM, self).__init__()
        self.cross_attention = CrossAttention(input_dim)
        self.fusion = CrossModalFusion(input_dim)
        self.fc1 = nn.Linear(hidden_dim * 80, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc3 = nn.Linear(hidden_dim // 4, num_classes)  # 输出二分类结果，所以最后一层输出维度为2
    def forward(self, feat, n, num_heads, head_dim, threshold1, threshold2):
        feat_brain = feat[:116, :]
        feat_gene = feat[116:, :]
        # 创建卷积模型实例
        cnn = CNN(n)
        # Numpy 数组转换为张量
        # feat_brain_tensor = torch.tensor(feat_brain, dtype=torch.float32)
        feat_brain_tensor = feat_brain.clone().detach()
        # feat_gene_tensor = torch.tensor(feat_gene, dtype=torch.float32)
        feat_gene_tensor = feat_gene.clone().detach()
        # 前向传播
        output1 = cnn(feat_brain_tensor.unsqueeze(0).unsqueeze(0))
        output2 = cnn(feat_gene_tensor.unsqueeze(0).unsqueeze(0))
        Mul_attention = Multiheads_Attention(output1, num_heads, head_dim, threshold1)
        Self_attention = Self_Attention(output2, threshold2)
        # 使用池化层池化Q1,K2向量
        Q1_pooled, K1_pooled, V1_pooled = self.cross_attention(Mul_attention)
        Q2_pooled, K2_pooled, V2_pooled = self.cross_attention(Self_attention)
        # 使用注意力机制处理池化后的向量
        attention_output = self.fusion(Q1_pooled, K1_pooled, V1_pooled, Q2_pooled, K2_pooled, V2_pooled)
        flattened_tensor = attention_output.view(1, -1)
        x = F.relu(self.fc1(flattened_tensor))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # 不使用 Softmax，因为在交叉熵损失函数中会自动处理
        return x, attention_output
