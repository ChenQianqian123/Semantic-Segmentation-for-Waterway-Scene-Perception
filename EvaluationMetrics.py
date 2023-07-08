import numpy as np
import os, time
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.metrics import *


'''
函数作用:本函数的作用主要是用来计算模型预测结果的准确性
主要包括：Accuracy, Precision, Recall, F1, IOU, mean IoU, FWIOU
实现思路：将所有分类结果分批次读入,拉成一维,计算单张图片的混淆矩阵
         然后将所有混淆矩阵结果叠加,得到总的混淆矩阵,计算后续精度
'''
class ImageOperation:

    def __init__(self):
        pass

    @staticmethod
    def openImage(image):
        return Image.open(image, mode="r")

    @staticmethod
    def saveImage(image, path):
        image.save(path)

def CalculatingConfusionMatrix(ture_path, predict_path, classes):
    '''
        P   N  真实值
    P  TP   FP

    N  FN   TN
    预测值
    '''
    # ture label path
    if os.path.isdir(ture_path):
        ture_label_names = os.listdir(ture_path)
    else:
        ture_label_names = [ture_path]

    # predict label path
    if os.path.isdir(predict_path):
        predict_label_names = os.listdir(predict_path)
    else:
        predict_label_names = [predict_path]

    ture_label_num = 0
    predcit_label_num = 0

    # statistics ture label num
    for ture_label_name in ture_label_names:
        tmp_ture_label_name = os.path.join(ture_path, ture_label_name)
        if os.path.isdir(tmp_ture_label_name):
            print('contain file folder')
            exit()
        else:
            ture_label_num = ture_label_num + 1;
    # statistics predcit label num
    for predict_label_name in predict_label_names:
        tmp_predict_label_name = os.path.join(predict_path, predict_label_name)
        if os.path.isdir(tmp_predict_label_name):
            print('contain file folder')
            exit()
        else:
            predcit_label_num = predcit_label_num + 1
    
    # ture label num == predcit label num
    if ture_label_num != predcit_label_num:
        print('the num of true label and predict label is not equl')
        exit()
    else:
        num = predcit_label_num

    confusion_matrixs = np.zeros((len(classes), len(classes)), np.uint8)

    sum_precision = 0
    sum_recall = 0
    sum_thresholds = 0

    # read all ture label and predict label
    for i in range(num):

        ture_label_name = ture_label_names[i]
        predict_label_name = predict_label_names[i]

        tmp_ture_label_name = os.path.join(ture_path, ture_label_name)
        tmp_predict_label_name = os.path.join(predict_path, predict_label_name)
        
        # read label and to array
        ture_label = np.asarray(ImageOperation.openImage(tmp_ture_label_name))
        predict_label = np.asarray(ImageOperation.openImage(tmp_predict_label_name))

        # calculating confusion matrix
        signle_confusion_matrix = confusion_matrix(ture_label.flatten(), predict_label.flatten(), classes)
        confusion_matrixs = confusion_matrixs + signle_confusion_matrix
    
    return confusion_matrixs

def CalculatingAccuracy(confusion_matrixs):
    '''
    Accuracy = (TP + TN) / (TP + FP + FN + TN)
    '''
    return np.diag(confusion_matrixs).sum() / confusion_matrixs.sum()

def CalculatingPrecision(confusion_matrixs):
    '''
    Precision = TP / (TP + FP)
    '''
    return np.diag(confusion_matrixs) / confusion_matrixs.sum(axis = 0)

def CalculatingRecall(confusion_matrixs):
    '''
    Recall = TP / (TP + FN)
    '''
    return np.diag(confusion_matrixs) / confusion_matrixs.sum(axis = 1)

def CalculatingF1Score(confusion_matrixs):
    '''
    F1Score = (2 * Precision * Recall) / (Precision + Recall)
    '''
    Precision = np.diag(confusion_matrixs) / confusion_matrixs.sum(axis = 0)
    Recall = np.diag(confusion_matrixs) / confusion_matrixs.sum(axis = 1)
    F1Score = (2 * Precision * Recall) / (Precision + Recall)

    return F1Score

def CalculatingIoU(confusion_matrixs):
    '''
    IoU = TP / (TP + FN + FP)
    '''
    intersection = np.diag(confusion_matrixs)  
    union = np.sum(confusion_matrixs, axis = 1) + np.sum(confusion_matrixs, axis = 0) - np.diag(confusion_matrixs) 
    IoU = intersection / union

    return IoU

def CalculatingMeanIoU(confusion_matrixs):
    '''
    MeanIoU = sum(IoU) / len(IoU)
    '''
    intersection = np.diag(confusion_matrixs)  
    union = np.sum(confusion_matrixs, axis = 1) + np.sum(confusion_matrixs, axis = 0) - np.diag(confusion_matrixs) 
    IoU = intersection / union
    MeanIoU = np.nanmean(IoU) 

    return MeanIoU

def CalculatingFWIoU(confusion_matrixs):
    '''
    FWIoU = (TP + FN)/(TP + FP + TN + FN) * (TP) / (TP + FP + FN)
    '''
    freq = np.sum(confusion_matrixs, axis=1) / np.sum(confusion_matrixs)  
    iu = np.diag(confusion_matrixs) / (
            np.sum(confusion_matrixs, axis = 1) +
            np.sum(confusion_matrixs, axis = 0) -
            np.diag(confusion_matrixs))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()

    return FWIoU


def AccuracyAssessment(ture_path, predict_path, classes):

    # read label and return confusion matrixs of adding all label
    confusion_matrixs = CalculatingConfusionMatrix(ture_path, predict_path, classes)
    # Accuracy
    Accuracy = CalculatingAccuracy(confusion_matrixs)
    # Precision
    Precision = CalculatingPrecision(confusion_matrixs)
    # Recall
    Recall = CalculatingRecall(confusion_matrixs)
    # F1
    F1Score = CalculatingF1Score(confusion_matrixs)
    # IOU
    IoU = CalculatingIoU(confusion_matrixs)
    # mean IoU
    MeanIoU = CalculatingMeanIoU(confusion_matrixs)
    # FWIOU
    FWIOUValue = CalculatingFWIoU(confusion_matrixs)

    return confusion_matrixs, Accuracy, Precision, Recall, F1Score, IoU, MeanIoU, FWIOUValue


if __name__ == '__main__':

    classes = [0, 1] # 所有的类别【0，1】表示有两类
    confusion_matrixs, Accuracy, Precision, Recall, F1Score, IoU, MeanIoU, FWIOUValue = AccuracyAssessment(
        "D:/My paper/NO/plot/3D/test3result/error/labelvalue/",#标准分割
        "D:/My paper/NO/plot/3D/test3result/error/Kmeansvalue/",#测试结果
        classes
    )
    print("混淆矩阵：")
    print(confusion_matrixs)
    print("Accuracy：")
    print(Accuracy)
    print("Precision：")
    print(Precision)
    print("Recall：")
    print(Recall)
    print("F1Score：")
    print(F1Score)
    print("IoU：")
    print(IoU)
    print("MeanIoU：")
    print(MeanIoU)
    print("FWIOUValue：")
    print(FWIOUValue)


