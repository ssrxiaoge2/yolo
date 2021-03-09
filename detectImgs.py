# !/usr/bin/python3
# -*-coding:utf-8-*-
import os
from yolo import *

yolo_dir = 'E:/Code/python/pythonProject/driving/yolo'  # YOLO文件路径
img_dir = 'E:/Code/python/pythonProject/driving/img'
weightsPath = os.path.join(yolo_dir, 'yolov3.weights')  # 权重文件
configPath = os.path.join(yolo_dir, 'yolov3.cfg')  # 配置文件
labelsPath = os.path.join(yolo_dir, 'driving.names')  # label名称
save_dir = 'save'
CONFIDENCE = 0.7  # 过滤弱检测的最小概率
THRESHOLD = 0.7  # 非最大值抑制阈值

net = getNet(configPath, weightsPath)

list_img = os.listdir(img_dir)

for filename in list_img:
    imgPath = os.path.join(img_dir, filename)  # 测试图像
    save_path = os.path.join(save_dir, filename)
    img = cv2.imread(imgPath)
    boxes, confidences, classIDs, idxs = detectionNpImg(img, net, (608, 608), CONFIDENCE, THRESHOLD)

    img = rectangle(img, labelsPath, boxes, confidences, classIDs, idxs)
    cv2.imwrite(save_path, img)

