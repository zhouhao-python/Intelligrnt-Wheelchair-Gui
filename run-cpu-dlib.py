# coding=utf-8
# -*- coding: utf-8 -*-
import torch
import serial.tools.list_ports
import torch.nn as nn
import os
import cv2
from models import create_dataloader,Classification,CnnLstm
from option import BasciOption
import dlib
import numpy as np
import time
from models import GoogleNet,GoogleNet3
from models import create_transform
import sys
# --------------------------------------------------------
# Camera sample code for Tegra X2/X1
#

# This program could capture and display 

# IP CAM, USB webcam, or the Tegra onboard camera.
# Refer to the following blog post for how to set up
# and run the code:
#   https://jkjung-avt.github.io/tx2-camera-with-python/
#
# Written by JK Jung <jkjung13@gmail.com>
# --------------------------------------------------------





def run_eye_detect():
    # show_camera_image()
    label = 'None'
    a,w,dd = 0,0,0
    use_cpu_dlib =  True
    far_count = 0
    near_count = 0
    plist = list(serial.tools.list_ports.comports())
    print(plist)
    if len(plist) <= 0:
        print("没有发现端口!")
    else:
        plist_0 = list(plist[1])
        serialName = plist_0[0]
        serialFd = serial.Serial(serialName, 115200, timeout=60)
        print("可用端口名>>>", serialFd.name)

        opt = BasciOption(train_flag=False).initialize()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("device is ", device)
        model = GoogleNet3()
        model.load_state_dict(torch.load(r"./Basic_Epoch_3_Accuracy_0.93.pth"))
        #近中远
        model = model.to(device)
        transformer = create_transform(opt)


        cap = cv2.VideoCapture(0)
        cap.set(3,640)
        cap.set(4,480)
        print(f'camera width = {cap.get(3)}\ncamera height = {cap.get(4)}')

        video_size = (200, 50)  # w h
        # offset_up_pixelY = -10
        # offset_bottom_pixelY = 5
        # offset_left_pixelX = 0
        # offset_right_pixelX = 0
        offset_pixelY = -10
        #offset_pixe2Y = 10
        offset_pixelX = 0
        #offset_pixe2X = 0
        eye_w,eye_h = 40,30


        predictor_path = "./dlib/shape_predictor_68_face_landmarks.dat"
        predictor = dlib.shape_predictor(predictor_path)

        # 初始化dlib人脸检测器
        if use_cpu_dlib == True:
            detector = dlib.get_frontal_face_detector()
        else:
            detector = dlib.cnn_face_detection_model_v1("./mmod_human_face_detector.dat")
        fps = 0.0
        face_offset = 100
        read_flag, _ = cap.read()
        while read_flag:
            pt_pos = []
            t1 = time.time()
            _, frame = cap.read()
            dets = detector(frame, 0)
            temp = len(dets)
            #print(type(dets))
            print(temp)
            print('number of faces detected: {}'.format(len(dets)))
            if temp == 0:
                serialFd.write('r'.encode())
                print("stop")
            if dets:
                for i, d in enumerate(dets):
                    if use_cpu_dlib == True:
                        shape = predictor(frame, d)  # 预测人脸形状大小
                    else:
                        # print('before',d.rect)
                        face = d.rect
                        left =face.left()
                        top = face.top()
                        right = face.right()
                        bottom = face.bottom()
                        d = dlib.rectangle(left,top,right,bottom)
                        # print('after',d)
                        cv2.rectangle(frame, (left,top), (right,bottom), (0,255,0), 3)
                    shape = predictor(frame, d)  # 预测人脸形状大小
                    for index, pt in enumerate(shape.parts()):
                        pt_pos.append((pt.x, pt.y))  # 人脸坐标点
                    left_eye = frame[pt_pos[37][1] + offset_pixelY:pt_pos[37][1] + eye_h+offset_pixelY,
                                     pt_pos[36][0] + offset_pixelX:pt_pos[36][0] +eye_w+ offset_pixelX]

                    right_eye = frame[pt_pos[44][1] + offset_pixelY:pt_pos[44][1] + eye_h+offset_pixelY,
                                      pt_pos[42][0] + offset_pixelX:pt_pos[42][0] + eye_w+offset_pixelX]
                    print("left_eye",left_eye.shape)
                    print("right_eye",right_eye.shape)
                    crop_eye = np.concatenate((left_eye, right_eye),
                                              axis=1)

                    crop_eye = cv2.resize(crop_eye, video_size)
                    crop_eye_show = cv2.resize(crop_eye,(600,200))
                    cv2.imshow('crop_eyes', crop_eye_show)
                    inputs = transformer(crop_eye).to(device)

                    outputs = model(inputs.unsqueeze(0))
                    _,y_pred = torch.max(outputs,dim = 1)
                    y_pred = y_pred.item()
                    print(y_pred)
                    if y_pred == 1:
                        w +=1
                        if w%5 == 0:
                            serialFd.write('r'.encode())
                            serialFd.write('w'.encode())
                    if y_pred == 0:
                        a +=1
                        if a%5 == 0:
                            serialFd.write('r'.encode())
                            serialFd.write('a'.encode())
                    if y_pred == 2:
                        dd +=1
                        if dd%5 == 0:
                            serialFd.write('r'.encode())
                            serialFd.write('d'.encode())
                    #if y_pred == 0:
                    #    near_count = 0
                    #    far_count += 1
                    #    
                    #    if far_count > 10: 
                    #        labal = 'Far'
                    #        serialFd.write('w'.encode())
                    #elif y_pred == 1:
                    #    far_count = 0
                    #    near_count += 1
                    #    
                    #    if near_count > 10000: 
                    #        label = "Near"
                    #        serialFd.write('s'.encode())
                    #else:
                    #    label = 'Mid'
                    #    serialFd.write('w'.encode())
            # print(frame)
                    fps = (fps + (1. / (time.time() - t1))) / 2
                # try:
                    #frame = cv2.putText(frame, f"fps={fps:5.3f}----{label}", (0, 100),
                     #                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    frame = cv2.putText(frame, f"fps={fps:5.3f}", (0, 100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                # except:
                #     continue
            frame = cv2.resize(frame,(1200,1200))
            cv2.imshow('origin_pic', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    run_eye_detect()