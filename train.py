import os
import utils_graph
import torch
import torch.optim as optim
from models import MWSAFM
import torch.nn as nn
import numpy as np
from freq_statistic import fre_statis
import torch.utils.data as Data

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 不用gpu，cuda有点问题

train_feat, train_label,test_feat, test_label = utils_graph.load_data()
# 每个批次的样本数
batch_size = 20
dataset = Data.TensorDataset(train_feat, train_label)
train_loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
model = MWSAFM(70, 70, 2)
# 卷积的次数
n = 5
# 定义头数和每个头的维度
num_heads = 5
head_dim = 14
LR = 0.00001
Mul_threshold = 0.5
Self_threshold = 0.5
EPOCH = 101
max_acc = 0
loss_list = []
acc_list = []
out_data = torch.zeros(420, 2)

optimizer = optim.Adam(model.parameters(), lr=LR)  # 优化器
loss_func = nn.CrossEntropyLoss()
for epoch in range(EPOCH):
    for step, (b_feat, b_lab) in enumerate(train_loader):
        outputs1 = []
        for feat in b_feat:
            output,_ = model(feat,n,num_heads, head_dim,Mul_threshold, Self_threshold)
            outputs1.append(output)
        result1 = torch.cat(outputs1, dim=0)
        loss = loss_func(result1, b_lab)
        acc_val = utils_graph.accuracy(result1, b_lab)
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        optimizer.step() # 更新模型参数

        if epoch == EPOCH - 1:
            if result1.shape[0] == batch_size:
                out_data[step * 20:(step + 1) * result1.shape[0], :] = result1

    if epoch % 10 == 0:
        print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.4f}'.format(loss.item()),
              'acc_val: {:.4f}'.format(acc_val))

        model.eval()
        outputs2 = []
        for feat in test_feat:
            output,_ = model(feat, n, num_heads, head_dim,Mul_threshold, Self_threshold)
            outputs2.append(output)
        result2 = torch.cat(outputs2, dim=0)
        loss_val1 = nn.CrossEntropyLoss()(result2, test_label)
        acc_val1 = utils_graph.accuracy(result2, test_label)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Test set results:",
              "loss= {:.4f}".format(loss_val1.item()),
              "accuracy= {:.4f}".format(acc_val1))
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        loss_list.append(float(loss_val1.item()))
        acc_list.append(float(acc_val1))
        model.train()
    if max_acc < acc_val1:
        max_acc = acc_val1
        ACC, SPE, SEN, PRE, AUC, F1 = utils_graph.stastic_indicators(result2, test_label)
        output2 = result2

# 特征提取
fc1_w = model.state_dict()['fc1.weight']
fc2_w = model.state_dict()['fc2.weight']
fc3_w = model.state_dict()['fc3.weight']
fc4 = out_data
best_imp, best_idx = fre_statis(fc1_w, fc2_w, fc3_w, fc4)
for num in range(best_idx.shape[0]):
    if fc4[num, 0] < fc4[num, 1]:
        best_index = best_idx[num]
        best_important = best_imp[num]
        break
#保存值
np.save('acc_list.npy',acc_list)
np.save('ACC.npy',ACC)
np.save('SPE.npy',SPE)
np.save('SEN.npy',SEN)
np.save('PRE.npy',PRE)
np.save('AUC.npy',AUC)
np.save('F1.npy',F1)
# np.save('TPR.npy',tpr)
# np.save('FPR.npy',fpr)
np.save('best_imp.npy',best_imp)
np.save('best_idx.npy',best_idx)