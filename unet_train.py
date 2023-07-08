import torch
import numpy as np
import os
import tensorflow as tf
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.transforms as transforms
from datetime import datetime
from utils.data_utils import Mydataset
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from unet import UNet
from torch.autograd import Variable
from sklearn.metrics import accuracy_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

class args:
    
    train_path = 'D:/My paper/NO/plot/3D/UNetSegmentation/UNetSegmentation/Creat Datasets/train_path_list.csv'
    val_path = 'D:/My paper/NO/plot/3D/UNetSegmentation/UNetSegmentation/Creat Datasets/val_path_list.csv'
    result_dir = 'D:/My paper/NO/plot/3D/UNetSegmentation/Unet/Result/'
    batch_size = 1
    learning_rate = 0.001
    max_epoch = 30

if __name__ == "__main__":

    best_train_acc = 0.80 # 当训练模型精度大于该值,保存当前模型

    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')

    log_dir = os.path.join(args.result_dir, time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir)

    #---------------------------1、加载数据---------------------------
    normMean = [0.56240946, 0.56267464, 0.5378807] # 统计不同波段的均值和方差进行归一化处理, 在compute_mean.py中统计
    normStd = [0.19984962, 0.19737755, 0.18974084]
    normTransfrom = transforms.Normalize(normMean, normStd)
    transform = transforms.Compose([
            transforms.ToTensor(),
            normTransfrom,
        ]) # 对数据转tensor,再对其进行归一化[-1, 1]
    # 构建Mydataset实例
    train_data = Mydataset(path = args.train_path, transform = transform)
    val_data = Mydataset(path = args.val_path, transform = transform)
    #构建DataLoader
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)

    #---------------------------2、定义网络---------------------------
    net = UNet(nInputChannels=3, n_classes=2, bilinear=False)  # 多类训练, n_classes 值修改即可
    net.cuda()
    # 将网络结构图传入tensorboard
    init_img = torch.randn((1, 3, 256, 256), device = 'cuda')
    writer.add_graph(net, init_img)

    #---------------------------3、初始化预训练权重、定义损失函数、优化器、设置超参数、---------------------------
    if torch.cuda.is_available(): # 类别权重用于计算损失函数
        w = torch.Tensor([1, 1]).cuda()
    else:
        w = torch.Tensor([1, 1])
    criterion = nn.CrossEntropyLoss(weight = w).cuda()
    optimizer = optim.SGD(net.parameters(), lr = args.learning_rate, momentum = 0.9, dampening = 0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.5)

    #---------------------------4、训练网络---------------------------
    for epoch in range(args.max_epoch):
        loss_sigma = 0.0
        acc_sigma = 0.0
        loss_val_sigma = 0.0
        acc_val_sigma = 0.0
        net.train()
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            labels = labels.long().cuda()
            optimizer.zero_grad()
            outputs = net.forward(inputs)
            loss = criterion(outputs, labels)
            predicts = torch.argmax(outputs, dim = 1) # softmax和argmax的区别
            acc_train = accuracy_score(np.reshape(labels.cpu(), [-1]), np.reshape(predicts.cpu(), [-1]))
            loss.backward()
            optimizer.step()
            # 统计预测信息
            loss_sigma += loss.item()
            acc_sigma += acc_train
            if i % 10 == 9:
                loss_avg = loss_sigma / 10
                acc_avg = acc_sigma / 10
                loss_sigma = 0.0
                acc_sigma = 0.0
                tf.compat.v1.logging.info("Training:Epoch[{:0>3}/{:0>3}] Iter[{:0>3}/{:0>3}] Loss:{:.4f} Acc:{:.4f}".format(
                    epoch + 1, args.max_epoch,i+1,len(train_loader),loss_avg,acc_avg))
                writer.add_scalar("LOSS", loss_avg, epoch)
                writer.add_scalar("ACCURACY", acc_avg, epoch)
                writer.add_scalar("LEARNING_RATE", optimizer.param_groups[0]["lr"], epoch)
                # 保存模型
                if (acc_avg) > best_train_acc:
                    # 保存精度最高的模型
                    net_save_path = os.path.join(log_dir, 'net_params.pkl')
                    torch.save(net.state_dict(), net_save_path)
                    best_train_acc = acc_avg
                    tf.compat.v1.logging.info('Save model successfully to "%s"!' % (log_dir + 'net_params.pkl'))

        net.eval()
        for i, data in enumerate(val_loader):
            inputs, labels = data
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            labels = labels.long().cuda()
            with torch.no_grad():
                outputs = net.forward(inputs)
            predicts = torch.argmax(outputs, dim=1)
            acc_val = accuracy_score(np.reshape(labels.cpu(), [-1]), np.reshape(predicts.cpu(), [-1]))
            loss_val = criterion(outputs, labels)
            # 统计预测信息
            loss_val_sigma += loss_val.item()
            acc_val_sigma += acc_val
        tf.compat.v1.logging.info("After 1 epoch：acc_val:{:.4f},loss_val:{:.4f}".format(acc_val_sigma/(len(val_loader)), loss_val_sigma/(len(val_loader))))
        acc_val_sigma = 0.0
        loss_val_sigma = 0.0
        scheduler.step() 

    writer.close()
    net_save_path = os.path.join(log_dir,'net_params_end.pkl')
    torch.save(net.state_dict(),net_save_path)