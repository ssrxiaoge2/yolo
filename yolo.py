# !/usr/bin/python3
# -*-coding:utf-8-*-
import numpy as np
import cv2
import time


def getNet(config, weights):
    """
    获取darknet网络
    :param config: yolo配置文件
    :param weights: 权重
    :return: 网络
    """
    return cv2.dnn.readNetFromDarknet(config, weights)


def detectionImg(imgPath, net, size=(416, 416), CONFIDENCE=0.5, THRESHOLD=0.4):
    # 加载图片、转为blob格式、送入网络输入层
    img = cv2.imread(imgPath)
    # net需要的输入是blob格式的，用blobFromImage这个函数来转格式
    blobImg = cv2.dnn.blobFromImage(img, 1.0 / 255.0, size, None, True, False)
    net.setInput(blobImg)  # 调用setInput函数将图片送入输入层

    # 获取网络输出层信息（所有输出层的名字），设定并前向传播
    # 前面的yolov3架构也讲了，yolo在每个scale都有输出，outInfo是每个scale的名字信息，供net.forward使用
    outInfo = net.getUnconnectedOutLayersNames()
    start = time.time()
    layerOutputs = net.forward(outInfo)  # 得到各个输出层的、各个检测框等信息，是二维结构。
    end = time.time()
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))  ## 可以打印下信息

    # 拿到图片尺寸
    (H, W) = img.shape[:2]

    # 过滤layerOutputs
    # layerOutputs的第1维的元素内容: [center_x, center_y, width, height, objectness, N-class score data]
    # 过滤后的结果放入：
    boxes = []  # 所有边界框（各层结果放一起）
    confidences = []  # 所有置信度
    classIDs = []  # 所有分类ID

    # 过滤掉置信度低的框框
    for out in layerOutputs:  # 各个输出层
        for detection in out:  # 各个框框
            # 拿到置信度
            scores = detection[5:]  # 各个类别的置信度
            classID = np.argmax(scores)  # 最高置信度的id即为分类id
            confidence = scores[classID]  # 拿到置信度

            # 根据置信度筛查
            if confidence > CONFIDENCE:
                box = detection[0:4] * np.array([W, H, W, H])  # 将边界框放会图片尺寸
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])  # 转变成左上角的点坐标+宽高
                confidences.append(float(confidence))
                classIDs.append(classID)
    # 2）应用非最大值抑制(non-maxima suppression，nms)进一步筛掉
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)  # boxes中，保留的box的索引index存入idxs

    return boxes, confidences, classIDs, idxs


def detectionNpImg(img, net, size=(416, 416), CONFIDENCE=0.5, THRESHOLD=0.4):
    # 加载图片、转为blob格式、送入网络输入层
    # net需要的输入是blob格式的，用blobFromImage这个函数来转格式
    blobImg = cv2.dnn.blobFromImage(img, 1.0 / 255.0, size, None, True, False)
    net.setInput(blobImg)  # 调用setInput函数将图片送入输入层

    # 获取网络输出层信息（所有输出层的名字），设定并前向传播
    # 前面的yolov3架构也讲了，yolo在每个scale都有输出，outInfo是每个scale的名字信息，供net.forward使用
    outInfo = net.getUnconnectedOutLayersNames()
    start = time.time()
    layerOutputs = net.forward(outInfo)  # 得到各个输出层的、各个检测框等信息，是二维结构。
    end = time.time()
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))  ## 可以打印下信息

    # 拿到图片尺寸
    (H, W) = img.shape[:2]

    # 过滤layerOutputs
    # layerOutputs的第1维的元素内容: [center_x, center_y, width, height, objectness, N-class score data]
    # 过滤后的结果放入：
    boxes = []  # 所有边界框（各层结果放一起）
    confidences = []  # 所有置信度
    classIDs = []  # 所有分类ID

    # 过滤掉置信度低的框框
    for out in layerOutputs:  # 各个输出层
        for detection in out:  # 各个框框
            # 拿到置信度
            scores = detection[5:]  # 各个类别的置信度
            classID = np.argmax(scores)  # 最高置信度的id即为分类id
            confidence = scores[classID]  # 拿到置信度

            # 根据置信度筛查
            if confidence > CONFIDENCE:
                box = detection[0:4] * np.array([W, H, W, H])  # 将边界框放会图片尺寸
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])  # 转变成左上角的点坐标+宽高
                confidences.append(float(confidence))
                classIDs.append(classID)
    # 2）应用非最大值抑制(non-maxima suppression，nms)进一步筛掉
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)  # boxes中，保留的box的索引index存入idxs

    return boxes, confidences, classIDs, idxs


def rectangle(img, labelsPath, boxes, confidences, classIDs, idxs):
    # 得到labels列表
    with open(labelsPath, 'rt') as f:
        labels = f.read().rstrip('\n').split('\n')

    # 应用检测结果
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(labels), 3),
                               dtype="uint8")  # 框框显示颜色，每一类有不同的颜色，每种颜色都是由RGB三个值组成的，所以size为(len(labels), 3)
    if len(idxs) > 0:
        for i in idxs.flatten():  # indxs是二维的，第0维是输出层，所以这里把它展平成1维
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)  # 线条粗细为2px
            text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                        2)  # cv.FONT_HERSHEY_SIMPLEX字体风格、0.5字体大小、粗细2px

    return img
