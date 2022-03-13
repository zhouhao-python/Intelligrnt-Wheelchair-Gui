# -*- coding: utf-8 -*-

"""
Module implementing mainWindow.
"""
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMainWindow
from intelligent_whellchair import Ui_intelligentwheelchair
import torch
import serial.tools.list_ports
import cv2
from option import BasciOption
import dlib
import numpy as np
from models import GoogleNet3
from models import create_transform
import sys
from PyQt5.QtCore import QThread, pyqtSignal
import source_pyqt_rc
from imutils import face_utils
from scipy.spatial import distance
import time


class mainWindow(QMainWindow, Ui_intelligentwheelchair):
    """
    Class documentation goes here.
    """

    def __init__(self, parent=None):
        """
        Constructor

        @param parent reference to the parent widget
        @type QWidget
        """
        super(mainWindow, self).__init__()
        self.setupUi(self)
        self.begin.clicked.connect(self.buttonClicked)

    def buttonClicked(self):
        self.label_4.setText("初始化......"+"\n")
        cap_num = int(self.comboBox_camera.currentText())
        self.thread = Worker(cap_num)
        self.thread.sig.connect(self.update_classification)
        self.thread.sig_face.connect(self.face_image)
        self.thread.sig_eye.connect(self.eye_image)
        self.thread.sig_in.connect(self.update_init)
        self.thread.start()

    def update_init(self,in_text):
        if in_text == "kong":
            QtWidgets.QMessageBox.warning(self, "警告", "没有发现串口信息！")
        else:
            self.label_4.setText(str(in_text))

    def update_classification(self,text):
        if text == 1:
            self.right.setStyleSheet("border-image: url(:/1/Desktop/right1.png);")
            self.left.setStyleSheet("border-image: url(:/1/Desktop/left1.png);")
            self.up.setStyleSheet("border-image: url(:/1/Desktop/up2.png);")
            self.stop.setStyleSheet("border-image: url(:/1/Desktop/forword11.png);")
        if text == 0:
            self.up.setStyleSheet("border-image: url(:/1/Desktop/up1.png);")
            self.right.setStyleSheet("border-image: url(:/1/Desktop/right1.png);")
            self.left.setStyleSheet("border-image: url(:/1/Desktop/left2.png);")
            self.stop.setStyleSheet("border-image: url(:/1/Desktop/forword11.png);")
        if text == 2:
            self.up.setStyleSheet("border-image: url(:/1/Desktop/up1.png);")
            self.left.setStyleSheet("border-image: url(:/1/Desktop/left1.png);")
            self.right.setStyleSheet("border-image: url(:/1/Desktop/right2.png);")
            self.stop.setStyleSheet("border-image: url(:/1/Desktop/forword11.png);")
        if text == 3:
            self.up.setStyleSheet("border-image: url(:/1/Desktop/up1.png);")
            self.left.setStyleSheet("border-image: url(:/1/Desktop/left1.png);")
            self.right.setStyleSheet("border-image: url(:/1/Desktop/right1.png);")
            self.stop.setStyleSheet("border-image: url(:/1/Desktop/forword21.png);")

    def eye_image(self,image_eye):
        image = cv2.cvtColor(image_eye, cv2.COLOR_BGR2RGB)
        height, width, bytesPerComponent = image.shape
        bytesPerLine = bytesPerComponent * width
        q_image = QtGui.QImage(image.data, width, height, bytesPerLine,
                               QtGui.QImage.Format_RGB888).scaled(self.eye.width(), self.eye.height())
        self.eye.setPixmap(QtGui.QPixmap.fromImage(q_image))

    def face_image(self,image):
        # print("debug",image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, bytesPerComponent = image.shape
        bytesPerLine = bytesPerComponent * width
        q_image = QtGui.QImage(image.data, width, height, bytesPerLine,
                               QtGui.QImage.Format_RGB888).scaled(self.face.width(), self.face.height())
        self.face.setPixmap(QtGui.QPixmap.fromImage(q_image))

    def closeEvent(self, QCloseEvent):
        '''
        窗口关闭事件
        :param QCloseEvent:
        :return:
        '''
        reply = QtWidgets.QMessageBox.question(self, "确认", "确认退出吗?", QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:

            QCloseEvent.accept()
        else:
            QCloseEvent.ignore()

class Worker(QThread):
    sig_in = pyqtSignal(str)
    sig = pyqtSignal(int)
    sig_face = pyqtSignal(object)
    sig_eye = pyqtSignal(object)

    def __init__(self, cap_num,parent=None):
        super(Worker, self).__init__(parent)
        self.cap_num = cap_num
        self.stop_flag = False
        self.EYE_EAR_BEYOND_TIME = 10 # 这个值设置的越小，眨眼所需的时间间隔越小
        self.RIGHT_EYE_START = 37 - 1
        self.RIGHT_EYE_END = 42 - 1
        self.LEFT_EYE_START = 43 - 1
        self.LEFT_EYE_END = 48 - 1
        self.EYE_EAR = 0.14  # EAR阈值 眨眼判定的幅度越大，越容易判断
        self.use_cpu_dlib = True
        self.predictor_path = "./dlib/shape_predictor_68_face_landmarks.dat"
        self.predictor = dlib.shape_predictor(self.predictor_path)
        self.time_list = []
        self.cap = cv2.VideoCapture(self.cap_num)

    def eye_aspect_ratio(self,eye):  # 计算EAR
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def run(self):
        self.time_list = []

        blink_counter = False
        frame_counter = 0
        frame_counter_stop = 0
        blink_counter_stop = False

        a, w, dd = 0, 0, 0
        plist = list(serial.tools.list_ports.comports())

        if len(plist) <= 0:
            self.sig_in.emit("kong")
        else:
            plist_0 = list(plist[0])
            serialName = plist_0[0]
            serialFd = serial.Serial(serialName, 115200, timeout=60)
            # print("可用端口名>>>", serialFd.name)
            opt = BasciOption(train_flag=False).initialize()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # print("device is ", device)
            model = GoogleNet3()
            model.load_state_dict(torch.load("./weight/Basic_Epoch_3_Accuracy_0.93.pth"))
            # 近中远
            model = model.to(device)
            transformer = create_transform(opt)

            self.cap.set(3, 640)
            self.cap.set(4, 480)
            # print(f'camera width = {cap.get(3)}\ncamera height = {cap.get(4)}')
            offset_pixelY = -10
            offset_pixelX = 0
            eye_w, eye_h = 40, 30

            # 初始化dlib人脸检测器
            self.sig_in.emit("可用端口名>>>{}".format(serialFd.name)+"\n"+"Device is {}".format(device)+"\n"+f'Camera Width = {self.cap.get(3)}\nCamera Height = {self.cap.get(4)}')

            if self.use_cpu_dlib == True:
                detector = dlib.get_frontal_face_detector()
            else:
                detector = dlib.cnn_face_detection_model_v1("./mmod_human_face_detector.dat")
            fps = 0.0
            read_flag,_ = self.cap.read()
            while read_flag:
                pt_pos = []
                t1 = time.time()
                _, frame = self.cap.read()
                dets = detector(frame, 0)
                temp = len(dets)
                self.sig_in.emit('检测到的人脸个数: {}'.format(len(dets)))
                if temp == 0:
                    serialFd.write('r'.encode())
                    self.sig.emit(3)
                if dets:
                    for i, d in enumerate(dets):
                        if self.use_cpu_dlib == True:
                            shape = self.predictor(frame, d)  # 预测人脸形状大小
                        else:
                            face = d.rect
                            left = face.left()
                            top = face.top()
                            right = face.right()
                            bottom = face.bottom()
                            d = dlib.rectangle(left, top, right, bottom)
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
                        shape = self.predictor(frame, d)  # 预测人脸形状大小
                        points = face_utils.shape_to_np(shape)
                        leftEye = points[self.LEFT_EYE_START:self.LEFT_EYE_END + 1]  # 取出左眼对应的特征点
                        rightEye = points[self.RIGHT_EYE_START:self.RIGHT_EYE_END + 1]  # 取出右眼对应的特征点
                        # print("lefteye",leftEye)
                        leftEAR = self.eye_aspect_ratio(leftEye)  # 计算左眼EAR
                        rightEAR = self.eye_aspect_ratio(rightEye)  # 计算右眼EAR
                        ear = (leftEAR + rightEAR) / 2.0
                        print("ear",ear)
                        # leftEyeHull = cv2.convexHull(leftEye)  # 寻找轮廓
                        # rightEyeHull = cv2.convexHull(rightEye)
                        # cv2.drawContours(frame, [leftEyeHull], -1, (192, 255, 62), 1)  # 绘制轮廓
                        # cv2.drawContours(frame, [rightEyeHull], -1, (192, 255, 62), 1)
                        if ear < self.EYE_EAR:  # 如果EAR小于阈值，开始累计连续眨眼次数
                            frame_counter += 1
                        else:
                            if frame_counter >= self.EYE_EAR_BEYOND_TIME:  # 连续帧计数超过EYE_EAR_BEYOND_TIME的帧数时，累加超时次数（blink_counter+1）并给予提示警告
                                print("frame_counter",frame_counter)
                                blink_counter = True
                                frame_counter = 0
                        if blink_counter == True:
                            self.sig.emit(3)
                            self.stop_flag = True
                            blink_counter = False
                            while self.stop_flag:
                                serialFd.write('r'.encode())
                                stop_flag,_ = self.cap.read()
                                while stop_flag and self.stop_flag:
                                    _, frame_stop = self.cap.read()
                                    dets_stop = detector(frame_stop, 0)
                                    if dets_stop:
                                        for i, d in enumerate(dets_stop):
                                            if self.use_cpu_dlib == True:
                                                shape_stop = self.predictor(frame_stop, d)  # 预测人脸形状大小
                                            else:
                                                face = d.rect
                                                left = face.left()
                                                top = face.top()
                                                right = face.right()
                                                bottom = face.bottom()
                                                d = dlib.rectangle(left, top, right, bottom)
                                                cv2.rectangle(frame_stop, (left, top), (right, bottom), (0, 255, 0), 3)
                                            shape_stop = self.predictor(frame_stop, d)  # 预测人脸形状大小
                                            points = face_utils.shape_to_np(shape_stop)
                                            leftEye_stop = points[
                                                      self.LEFT_EYE_START:self.LEFT_EYE_END + 1]  # 取出左眼对应的特征点
                                            rightEye_stop = points[
                                                       self.RIGHT_EYE_START:self.RIGHT_EYE_END + 1]  # 取出右眼对应的特征点
                                            leftEAR_stop = self.eye_aspect_ratio(leftEye_stop)  # 计算左眼EAR
                                            rightEAR_stop = self.eye_aspect_ratio(rightEye_stop)  # 计算右眼EAR
                                            ear_stop = (leftEAR_stop + rightEAR_stop) / 2.0
                                            if ear_stop < self.EYE_EAR:  # 如果EAR小于阈值，开始累计连续眨眼次数
                                                frame_counter_stop += 1
                                            else:
                                                if frame_counter_stop >= self.EYE_EAR_BEYOND_TIME:  # 连续帧计数超过EYE_EAR_BEYOND_TIME的帧数时，累加超时次数（blink_counter+1）并给予提示警告
                                                    print("frame_counter", frame_counter)
                                                    blink_counter_stop = True
                                                    frame_counter_stop = 0
                                                    print("blink_counter stop", blink_counter_stop)
                                            if blink_counter_stop == True:
                                                self.stop_flag = False
                                                blink_counter_stop = False
                        for index, pt in enumerate(shape.parts()):
                            pt_pos.append((pt.x, pt.y))  # 人脸坐标点
                        left_eye = frame[pt_pos[37][1] + offset_pixelY:pt_pos[37][1] + eye_h + offset_pixelY,
                                   pt_pos[36][0] + offset_pixelX:pt_pos[36][0] + eye_w + offset_pixelX]

                        right_eye = frame[pt_pos[44][1] + offset_pixelY:pt_pos[44][1] + eye_h + offset_pixelY,
                                    pt_pos[42][0] + offset_pixelX:pt_pos[42][0] + eye_w + offset_pixelX]
                        try:
                            crop_eye = np.concatenate((left_eye, right_eye),
                                                  axis=1)
                            self.sig_eye.emit(crop_eye)
                        except:
                            continue
                        inputs = transformer(crop_eye).to(device)
                        outputs = model(inputs.unsqueeze(0))
                        _, y_pred = torch.max(outputs, dim=1)
                        y_pred = y_pred.item()
                        if y_pred == 1:
                            w += 1
                            if w % 5 == 0:
                                self.sig.emit(int(y_pred))
                                serialFd.write('r'.encode())
                                serialFd.write('w'.encode())
                        if y_pred == 0:
                            a += 1
                            if a % 5 == 0:
                                self.sig.emit(int(y_pred))
                                serialFd.write('r'.encode())
                                serialFd.write('a'.encode())
                        if y_pred == 2:
                            dd += 1
                            if dd % 5 == 0:
                                self.sig.emit(int(y_pred))
                                serialFd.write('r'.encode())
                                serialFd.write('d'.encode())
                        fps = (fps + (1. / (time.time() - t1))) / 2
                        frame = cv2.putText(frame, f"fps={fps:5.3f}", (0, 100),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                self.sig_face.emit(frame)
        self.cap.release()

    @pyqtSlot()
    def gui_show(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = mainWindow()
    ui.show()
    sys.exit(app.exec_())
