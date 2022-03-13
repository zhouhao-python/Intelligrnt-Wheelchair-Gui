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
from models import GoogleNet,GoogleNet3,GoogleNet9
from models import create_transform
import sys
# --------------------------------------------------------
# Camera sample code for Tegra X2/X1
#
# This program could capture and display video from
# IP CAM, USB webcam, or the Tegra onboard camera.
# Refer to the following blog post for how to set up
# and run the code:
#   https://jkjung-avt.github.io/tx2-camera-with-python/
#
# Written by JK Jung <jkjung13@gmail.com>
# --------------------------------------------------------

if __name__ == '__main__':
    # show_camera_image()
    
    plist = list(serial.tools.list_ports.comports())
    if len(plist) <= 0:
        print("没有发现端口!")
    else:
        plist_0 = list(plist[0])
        serialName = plist_0[0]
        serialFd = serial.Serial(serialName, 115200, timeout=60)
        print("可用端口名>>>", serialFd.name)
        while True:
            a = input("inputs:")
            print(type(a))
            if int(a) == 1:
                serialFd.write('w'.encode())
            elif int(a) == 2:
                serialFd.write('s'.encode())
            elif int(a) == 3:
                serialFd.write(str('w').encode())
            elif int(a) == 4:
                serialFd.write(str('s').encode())
        
