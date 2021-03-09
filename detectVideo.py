# !/usr/bin/python3
# -*-coding:utf-8-*-
# 识别视频
import numpy as np
import cv2
import os
import time
from yolo import *

yolo_dir = 'C:/Users/t1467\Desktop/driving/yolo'  # YOLO文件路径
img_dir = 'C:/Users/t1467\Desktop/driving/img'
weightsPath = os.path.join(yolo_dir, 'yolov3-tiny.weights')  # 权重文件
configPath = os.path.join(yolo_dir, 'yolov3-tiny.cfg')  # 配置文件
labelsPath = os.path.join(yolo_dir, 'driving.names')  # label名称
videoPath = 'video/E.mp4'
CONFIDENCE = 0.7  # 过滤弱检测的最小概率
THRESHOLD = 0.7  # 非最大值抑制阈值

#读取配置文件和权重文件进行图片识别
net = getNet(configPath, weightsPath)

#读取视频文件videopath 为0则为调用摄像头
cap = cv2.VideoCapture(videoPath)

#获得视频的宽高
sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#视频帧率
fps = cap.get(cv2.CAP_PROP_FPS)
#参数：
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# fourcc = cv2.VideoWriter_fourcc(*'mpeg')

#获得一个视频输出流
out = cv2.VideoWriter()
#打开文件
out.open('./video/output.mp4', fourcc, fps, sz, True)
write_ok=False
print(fps)
while cap.isOpened():

    #return_value判断视频是否节输, img = cap.read()读取这一帧的图片
    return_value, img = cap.read()
    if return_value is True:
        #检测图片，获取返回值
        boxes, confidences, classIDs, idxs = detectionNpImg(img, net, (608, 608), CONFIDENCE, THRESHOLD)
        #画检测框
        img = rectangle(img, labelsPath, boxes, confidences, classIDs, idxs)
        #显示窗口实时展示
        cv2.imshow("Canvas", img)
        #输出视频文件
        out.write(img)
        #按q退出识别
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

#释放资源
out.release()
cap.release()
cv2.destroyAllWindows()
