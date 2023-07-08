import torch
import math
import os
import numpy as np
import cv2
import tensorflow as tf
import torch.nn.functional as F
import torchvision.transforms as transforms
from  utils.color_utils import color_annotation
from sklearn.metrics import accuracy_score
from unet import UNet
from utils.data_utils import Mydataset
from torch.utils.data import DataLoader
from scipy.io import savemat
'''
本代码需要修改地方
1. 26行,修改为你自己的模型存在路径 
2. 27行,修改为你自己的结果输出文件夹
3. 56行,修改为你自己的存储测试图片csv的路径
4. 209行,修改为你存储测试图片的文件夹,主要目的是获取文件名
'''


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

pretrained_path = 'D:/My paper/NO/plot/3D/UNetSegmentation/UNetSegmentation/Unet/Result/07-11_23-20-18/net_params_end.pkl' # 模型路径
output_path = 'D:/My paper/NO/plot/3D/UNetSegmentation/UNetSegmentation/Creat Datasets/dataset/experiment dataset/test3/predict/' # 分类结果输出文件夹
pro_path = 'D:/My paper/NO/plot/3D/UNetSegmentation/UNetSegmentation/Creat Datasets/dataset/experiment dataset/test3/probability/' # 概率结果输出文件夹
scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]  # 多尺度预测
OA_all = []
classes = 2 # 预测修改类别数
normMean = [0.56240946, 0.56267464, 0.5378807]
normStd = [0.19984962, 0.19737755, 0.18974084]
crop_h = 256
crop_w = 256

# 构建模型,载入训练好的权重参数
net = UNet(nInputChannels=3, n_classes=2, bilinear=False) # 预测修改类别数
net.eval()
if torch.cuda.is_available():
    #支持cuda计算的情况下
    net = net.cuda()
    pretrained_dict = torch.load(pretrained_path, map_location=torch.device('cuda'))
    net.load_state_dict(pretrained_dict)
else:
    # 不支持cuda计算的情况下
    pretrained_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    net.load_state_dict(pretrained_dict)

# 数据预处理
normTransfrom = transforms.Normalize(normMean, normStd)
transform = transforms.Compose([
    transforms.ToTensor(),
    normTransfrom,
])
# 构建Mydataset实例
test_data = Mydataset(path='D:/My paper/NO/plot/3D/UNetSegmentation/UNetSegmentation/Creat Datasets/test_path_list.csv', transform=transform) #  测试集路径文件
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

def net_process(model, image, flip=True):
    input = torch.from_numpy(image.transpose((2, 0, 1))).float()
    if torch.cuda.is_available():
        input = input.unsqueeze(0).cuda()
    else:
        input = input.unsqueeze(0)
    if flip:
        input = torch.cat([input, input.flip(3)], 0)
    with torch.no_grad():
        output = model(input)
    _, _, h_i, w_i = input.shape
    _, _, h_o, w_o = output.shape
    if (h_o != h_i) or (w_o != w_i):
        output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)
    output = F.softmax(output, dim=1)
    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]
    output = output.data.cpu().numpy()
    output = output.transpose(1, 2, 0)
    return output

def scale_process(model, image, classes, crop_h, crop_w, h, w, mean, stride_rate=2/3):
    ori_h, ori_w, _ = image.shape
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                   cv2.BORDER_CONSTANT, value=mean)
    new_h, new_w, _ = image.shape
    stride_h = int(np.ceil(crop_h * stride_rate))
    stride_w = int(np.ceil(crop_w * stride_rate))
    grid_h = int(np.ceil(float(new_h - crop_h) / stride_h) + 1)
    grid_w = int(np.ceil(float(new_w - crop_w) / stride_w) + 1)
    prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)
    count_crop = np.zeros((new_h, new_w), dtype=float)
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[s_h:e_h, s_w:e_w].copy()
            count_crop[s_h:e_h, s_w:e_w] += 1
            prediction_crop[s_h:e_h, s_w:e_w, :] += net_process(model, image_crop)
    prediction_crop /= np.expand_dims(count_crop, 2)
    prediction_crop = prediction_crop[pad_h_half:pad_h_half + ori_h, pad_w_half:pad_w_half + ori_w]
    prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return prediction

def TifCroppingArray(img, SideLength):
    #  tif裁剪（tif像素数据，裁剪边长）
    #  裁剪链表
    TifArrayReturn = []
    #  列上图像块数目
    ColumnNum = int((img.shape[0] - SideLength * 2) / (256 - SideLength * 2))
    #  行上图像块数目
    RowNum = int((img.shape[1] - SideLength * 2) / (256 - SideLength * 2))
    for i in range(ColumnNum):
        TifArray = []
        for j in range(RowNum):
            cropped = img[i * (256 - SideLength * 2) : i * (256 - SideLength * 2) + 256,
                          j * (256 - SideLength * 2) : j * (256 - SideLength * 2) + 256]
            TifArray.append(cropped)
        TifArrayReturn.append(TifArray)
    #  考虑到行列会有剩余的情况，向前裁剪一行和一列
    #  向前裁剪最后一列
    for i in range(ColumnNum):
        cropped = img[i * (256 - SideLength * 2) : i * (256 - SideLength * 2) + 256,
                      (img.shape[1] - 256) : img.shape[1]]
        TifArrayReturn[i].append(cropped)
    #  向前裁剪最后一行
    TifArray = []
    for j in range(RowNum):
        cropped = img[(img.shape[0] - 256) : img.shape[0],
                      j * (256-SideLength*2) : j * (256 - SideLength * 2) + 256]
        TifArray.append(cropped)
    #  向前裁剪右下角
    cropped = img[(img.shape[0] - 256) : img.shape[0],
                  (img.shape[1] - 256) : img.shape[1]]
    TifArray.append(cropped)
    TifArrayReturn.append(TifArray)
    #  列上的剩余数
    ColumnOver = (img.shape[0] - SideLength * 2) % (256 - SideLength * 2) + SideLength
    #  行上的剩余数
    RowOver = (img.shape[1] - SideLength * 2) % (256 - SideLength * 2) + SideLength

    return TifArrayReturn, RowOver, ColumnOver

def Result(shape, TifArray, npyfile, num_class, RepetitiveLength, RowOver, ColumnOver):
    #  获得结果矩阵
    result = np.zeros(shape, np.uint8)
    #  j来标记行数
    j = 0  
    for i, img in enumerate(npyfile):
        img = img.astype(np.uint8)
        #  最左侧一列特殊考虑，左边的边缘要拼接进去
        if(i % len(TifArray[0]) == 0):
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if(j == 0):
                result[0 : 256 - RepetitiveLength, 0 : 256-RepetitiveLength] = img[0 : 256 - RepetitiveLength, 0 : 256 - RepetitiveLength]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif(j == len(TifArray) - 1):
                result[shape[0] - ColumnOver - RepetitiveLength: shape[0], 0 : 256 - RepetitiveLength] = img[256 - ColumnOver - RepetitiveLength : 256, 0 : 256 - RepetitiveLength]
            else:
                result[j * (256 - 2 * RepetitiveLength) + RepetitiveLength : (j + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength,
                       0:256-RepetitiveLength] = img[RepetitiveLength : 256 - RepetitiveLength, 0 : 256 - RepetitiveLength]   
        #  最右侧一列特殊考虑，右边的边缘要拼接进去
        elif(i % len(TifArray[0]) == len(TifArray[0]) - 1):
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if(j == 0):
                result[0 : 256 - RepetitiveLength, shape[1] - RowOver: shape[1]] = img[0 : 256 - RepetitiveLength, 256 -  RowOver: 256]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif(j == len(TifArray) - 1):
                result[shape[0] - ColumnOver : shape[0], shape[1] - RowOver : shape[1]] = img[256 - ColumnOver : 256, 256 - RowOver : 256]
            else:
                result[j * (256 - 2 * RepetitiveLength) + RepetitiveLength : (j + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength,
                       shape[1] - RowOver : shape[1]] = img[RepetitiveLength : 256 - RepetitiveLength, 256 - RowOver : 256]   
            #  走完每一行的最右侧，行数+1
            j = j + 1
        #  不是最左侧也不是最右侧的情况
        else:
            #  第一行的要特殊考虑，上边的边缘要考虑进去
            if(j == 0):
                result[0 : 256 - RepetitiveLength,
                       (i - j * len(TifArray[0])) * (256 - 2 * RepetitiveLength) + RepetitiveLength : (i - j * len(TifArray[0]) + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength
                       ] = img[0 : 256 - RepetitiveLength, RepetitiveLength : 256 - RepetitiveLength]         
            #  最后一行的要特殊考虑，下边的边缘要考虑进去
            if(j == len(TifArray) - 1):
                result[shape[0] - ColumnOver : shape[0],
                       (i - j * len(TifArray[0])) * (256 - 2 * RepetitiveLength) + RepetitiveLength : (i - j * len(TifArray[0]) + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength
                       ] = img[256 - ColumnOver : 256, RepetitiveLength : 256 - RepetitiveLength]
            else:
                result[j * (256 - 2 * RepetitiveLength) + RepetitiveLength : (j + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength,
                       (i - j * len(TifArray[0])) * (256 - 2 * RepetitiveLength) + RepetitiveLength : (i - j * len(TifArray[0]) + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength,
                       ] = img[RepetitiveLength : 256 - RepetitiveLength, RepetitiveLength : 256 - RepetitiveLength]
    return result

def ProbabilityResult(shape, TifArray, npyfile, num_class, RepetitiveLength, RowOver, ColumnOver):
    #  获得结果矩阵
    result = np.zeros(shape, np.float)
    #  j来标记行数
    j = 0  
    for i, img in enumerate(npyfile):
        #  最左侧一列特殊考虑，左边的边缘要拼接进去
        if(i % len(TifArray[0]) == 0):
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if(j == 0):
                result[0 : 256 - RepetitiveLength, 0 : 256-RepetitiveLength, :] = img[0 : 256 - RepetitiveLength, 0 : 256 - RepetitiveLength, :]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif(j == len(TifArray) - 1):
                result[shape[0] - ColumnOver - RepetitiveLength: shape[0], 0 : 256 - RepetitiveLength, :] = img[256 - ColumnOver - RepetitiveLength : 256, 0 : 256 - RepetitiveLength, :]
            else:
                result[j * (256 - 2 * RepetitiveLength) + RepetitiveLength : (j + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength,
                       0:256-RepetitiveLength, :] = img[RepetitiveLength : 256 - RepetitiveLength, 0 : 256 - RepetitiveLength, :]   
        #  最右侧一列特殊考虑，右边的边缘要拼接进去
        elif(i % len(TifArray[0]) == len(TifArray[0]) - 1):
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if(j == 0):
                result[0 : 256 - RepetitiveLength, shape[1] - RowOver: shape[1], :] = img[0 : 256 - RepetitiveLength, 256 -  RowOver: 256, :]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif(j == len(TifArray) - 1):
                result[shape[0] - ColumnOver : shape[0], shape[1] - RowOver : shape[1], :] = img[256 - ColumnOver : 256, 256 - RowOver : 256, :]
            else:
                result[j * (256 - 2 * RepetitiveLength) + RepetitiveLength : (j + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength,
                       shape[1] - RowOver : shape[1], :] = img[RepetitiveLength : 256 - RepetitiveLength, 256 - RowOver : 256, :]   
            #  走完每一行的最右侧，行数+1
            j = j + 1
        #  不是最左侧也不是最右侧的情况
        else:
            #  第一行的要特殊考虑，上边的边缘要考虑进去
            if(j == 0):
                result[0 : 256 - RepetitiveLength,
                       (i - j * len(TifArray[0])) * (256 - 2 * RepetitiveLength) + RepetitiveLength : (i - j * len(TifArray[0]) + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength
                       , :] = img[0 : 256 - RepetitiveLength, RepetitiveLength : 256 - RepetitiveLength, :]         
            #  最后一行的要特殊考虑，下边的边缘要考虑进去
            if(j == len(TifArray) - 1):
                result[shape[0] - ColumnOver : shape[0],
                       (i - j * len(TifArray[0])) * (256 - 2 * RepetitiveLength) + RepetitiveLength : (i - j * len(TifArray[0]) + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength, :
                       ] = img[256 - ColumnOver : 256, RepetitiveLength : 256 - RepetitiveLength, :]
            else:
                result[j * (256 - 2 * RepetitiveLength) + RepetitiveLength : (j + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength,
                       (i - j * len(TifArray[0])) * (256 - 2 * RepetitiveLength) + RepetitiveLength : (i - j * len(TifArray[0]) + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength, :
                       ] = img[RepetitiveLength : 256 - RepetitiveLength, RepetitiveLength : 256 - RepetitiveLength, :]
    return result

if __name__ == '__main__':

    area_perc = 0.5
    RepetitiveLength = int((1 - math.sqrt(area_perc)) * 256 / 2) # 计算重复像素

    # 读取测试图像文件名
    testpath = 'D:/My paper/NO/plot/3D/UNetSegmentation/UNetSegmentation/Creat Datasets/dataset/experiment dataset/test2/image2/'
    # img path
    if os.path.isdir(testpath):
        img_names = os.listdir(testpath)
    else:
        img_names = [testpath]
    # 多尺度测试
    for i, data in enumerate(test_loader): # 按顺序索引
        input, label = data
        input = np.squeeze(input.numpy(), axis=0)
        traninput = np.transpose(input, (1, 2, 0)) # 对image进行裁剪再拼接
        TifArray, RowOver, ColumnOver = TifCroppingArray(traninput, RepetitiveLength) # TifArrayReturn为裁剪链表, RowOver, ColumnOver分别表示行列剩余数
        results = []
        probability = []
        for j in range(len(TifArray)):
            for k in range(len(TifArray[0])):
                image = TifArray[j][k]       
                h, w, _ = image.shape
                prediction = np.zeros((h, w, classes), dtype=float)
                for scale in scales:
                    base_size = 0
                    if h > w:
                        base_size = h
                    else:
                        base_size = w
                    long_size = round(scale * base_size) # 四舍五入
                    new_h = long_size
                    new_w = long_size
                    if h > w:
                        new_w = round(long_size / float(h) * w)
                    else:
                        new_h = round(long_size / float(w) * h)
                    image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    prediction += scale_process(net, image_scale, classes, crop_h, crop_w, h, w, normMean)
                prediction /= len(scales)
                probability.append(prediction) # 添加概率
                prediction = np.argmax(prediction, axis=2)
                results.append(prediction)  # 添加预测结果
        result_shape = (traninput.shape[0], traninput.shape[1])
        result_probabilityshape = (traninput.shape[0], traninput.shape[1], classes)
        result_data = Result(result_shape, TifArray, results, 2, RepetitiveLength, RowOver, ColumnOver)
        result_probability = ProbabilityResult(result_probabilityshape, TifArray, probability, 2, RepetitiveLength, RowOver, ColumnOver)
        result_probability = np.array(result_probability)
        result_probability = np.transpose(result_probability, (2, 0, 1))
        color_annotation(result_data, result_probability, output_path + img_names[i])
        A = result_probability[0, :, :]
        output_pro_path = pro_path + "/%05d.mat"%(i+1)
        savemat(output_pro_path, mdict={'pro':A})


