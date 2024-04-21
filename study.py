# -*- coding:utf-8 -*-
"""
作者：知行合一
日期：2019年 08月 21日 14:12
文件名：study.py
地点：changsha
"""
"""
根据训练好的yolov5模型，进行推理
"""
import cv2
import numpy as np
import time
import torch

# https://github.com/ultralytics/yolov5/issues/36

class PPE_detector:

    def __init__(self):
        # 加载模型
        self.model = torch.hub.load('./yolov5','custom',path='./weights/ppe_yolo_n.pt',source='local')
        self.model.conf = 0.4
        # 获取视频流
        self.cap = cv2.VideoCapture(0)

    def detect(self):
        # 获取视频的每一帧
        while True:
            ret,frame = self.cap.read()

            # 画面翻转
            frame = cv2.flip(frame,1)

            # 画面转为RGB格式
            frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            # 执行推理过程
            results = self.model(frame_rgb)

            results_np = results.pandas().xyxy[0].to_numpy()

            # print(results_np)

            # 绘制边界框
            for box in results_np:
                l,t,r,b = box[:4].astype('int')

                # 获取label_id
                label_id = box[5]
                if label_id == 0:
                    cv2.rectangle(frame, (l,t), (r,b) ,( 0, 255 , 0), 5)
                else:
                    cv2.rectangle(frame, (l, t), (r, b), (0, 255, 255), 5)


            # 显示画面
            cv2.imshow('PPE demo',frame)

            if cv2.waitKey(10) & 0xFF==ord('q'):
                break


        self.cap.release()
        cv2.destroyAllWindows()



ppe = PPE_detector() # 实例化
ppe.detect()
