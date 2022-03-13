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
import time
from models import GoogleNet3
from models import create_transform
import sys
from PyQt5.QtCore import QThread, pyqtSignal
import source_pyqt_rc


# from Ui_first import Ui_mainWindow


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


    # def video_play_slot(self):
    #     if not self.cap.isOpened():
    #         QtWidgets.QMessageBox.warning(self, "警告", "未成功打开摄像头！请关闭界面并重新打开！")
    #     else:
    #         if not self.video_timer.isActive():
    #             self.video_timer.start(600)
    #             self.video_timer.timeout.connect(self.run_eye_detect)
    #         else:
    #             self.captureImg_flag = True



    def buttonClicked(self):
        self.label_4.setText("init......"+"\n")
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
        if text == 0:
            self.up.setStyleSheet("border-image: url(:/1/Desktop/up1.png);")
            self.right.setStyleSheet("border-image: url(:/1/Desktop/right1.png);")
            self.left.setStyleSheet("border-image: url(:/1/Desktop/left2.png);")
        if text == 2:
            self.up.setStyleSheet("border-image: url(:/1/Desktop/up1.png);")
            self.left.setStyleSheet("border-image: url(:/1/Desktop/left1.png);")
            self.right.setStyleSheet("border-image: url(:/1/Desktop/right2.png);")

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
        # self.count = 0
        self.cap_num = cap_num
        # print("debug2",self.cap_num)

    def run(self):
        a, w, dd = 0, 0, 0
        use_cpu_dlib = True
        plist = list(serial.tools.list_ports.comports())
        # print(plist)
        if len(plist) <= 0:
            self.sig_in.emit("kong")
        else:
            plist_0 = list(plist[0])
            serialName = plist_0[0]
            serialFd = serial.Serial(serialName, 115200, timeout=60)
            print("可用端口名>>>", serialFd.name)
            opt = BasciOption(train_flag=False).initialize()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print("device is ", device)
            model = GoogleNet3()
            model.load_state_dict(torch.load(r"./Basic_Epoch_3_Accuracy_0.93.pth"))
            # 近中远
            model = model.to(device)
            transformer = create_transform(opt)
            cap = cv2.VideoCapture(self.cap_num)
            cap.set(3, 640)
            cap.set(4, 480)
            print(f'camera width = {cap.get(3)}\ncamera height = {cap.get(4)}')
            offset_pixelY = -10
            offset_pixelX = 0
            eye_w, eye_h = 40, 30
            predictor_path = "./dlib/shape_predictor_68_face_landmarks.dat"
            predictor = dlib.shape_predictor(predictor_path)
            # 初始化dlib人脸检测器
            self.sig_in.emit("可用端口名>>>{}".format(serialFd.name)+"\n"+"device is {}".format(device)+"\n"+f'camera width = {cap.get(3)}\ncamera height = {cap.get(4)}')

            if use_cpu_dlib == True:
                detector = dlib.get_frontal_face_detector()
            else:
                detector = dlib.cnn_face_detection_model_v1("./mmod_human_face_detector.dat")
            fps = 0.0
            read_flag, _ = cap.read()
            while read_flag:
                pt_pos = []
                t1 = time.time()
                _, frame = cap.read()
                dets = detector(frame, 0)
                temp = len(dets)
                # print(type(dets))
                # print(temp)
                print('number of faces detected: {}'.format(len(dets)))

                # if temp == 0:
                #     serialFd.write('r'.encode())
                #     print("stop")

                if dets:
                    for i, d in enumerate(dets):
                        if use_cpu_dlib == True:
                            shape = predictor(frame, d)  # 预测人脸形状大小
                        else:
                            face = d.rect
                            left = face.left()
                            top = face.top()
                            right = face.right()
                            bottom = face.bottom()
                            d = dlib.rectangle(left, top, right, bottom)
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
                        shape = predictor(frame, d)  # 预测人脸形状大小
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
                                # serialFd.write('r'.encode())
                                # serialFd.write('w'.encode())
                        if y_pred == 0:
                            a += 1
                            if a % 5 == 0:
                                self.sig.emit(int(y_pred))
                                # serialFd.write('r'.encode())
                                # serialFd.write('a'.encode())
                        if y_pred == 2:
                            dd += 1
                            if dd % 5 == 0:
                                self.sig.emit(int(y_pred))
                                # serialFd.write('r'.encode())
                                # serialFd.write('d'.encode())
                        fps = (fps + (1. / (time.time() - t1))) / 2
                        frame = cv2.putText(frame, f"fps={fps:5.3f}", (0, 100),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                self.sig_face.emit(frame)
        cap.release()
    @pyqtSlot()
    def gui_show(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    ui = mainWindow()
    ui.show()
    sys.exit(app.exec_())
