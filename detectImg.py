# !/usr/bin/python3
# -*-coding:utf-8-*-
import os
from yolo import *
#
yolo_dir = 'E:/Code/python/pythonProject/driving/yolo'  # YOLO文件路径
img_dir = 'E:/Code/python/pythonProject/driving/img'
weightsPath = os.path.join(yolo_dir, 'yolov3.weights')  # 权重文件
configPath = os.path.join(yolo_dir, 'yolov3.cfg')  # 配置文件
labelsPath = os.path.join(yolo_dir, 'driving.names')  # label名称
imgPath = os.path.join(img_dir, '1594547120301.jpg')  # 测试图像
CONFIDENCE = 0.7  # 过滤弱检测的最小概率
THRESHOLD = 0.7  # 非最大值抑制阈值

net = getNet(configPath, weightsPath)

img = cv2.imread(imgPath)
boxes, confidences, classIDs, idxs = detectionNpImg(img, net, (608, 608), CONFIDENCE, THRESHOLD)

img = rectangle(img, labelsPath, boxes, confidences, classIDs, idxs)
cv2.imshow("img", img)
cv2.waitKey(0)
