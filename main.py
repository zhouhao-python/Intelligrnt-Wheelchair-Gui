# -*- coding: utf-8 -*-

"""
Module implementing mainWindow.
"""
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot,QTimer,QDateTime,Qt
from PyQt5.QtWidgets import QMainWindow,QStackedWidget,QMessageBox
from PyQt5.QtGui import QIcon,QImage,QPixmap
from design.design import Ui_StackedWidget
import torch
import serial.tools.list_ports
import cv2
import dlib
import numpy as np
from models import GoogleNet3
from models import create_transform
import sys
from PyQt5.QtCore import QThread, pyqtSignal,QSettings
from imutils import face_utils
from scipy.spatial import distance
import time
from drawlabel import Drawlabel
from PyQt5.QtSerialPort import QSerialPort
from series import Series

class mainWindow(QStackedWidget, Ui_StackedWidget):
    main_to_thread = pyqtSignal(list)
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
        self.showtimer = QTimer()
        self.showtimer.timeout.connect(self.showtime)
        self.showtimer.start(1000)
        self.setimage()
        self.buttonClicked()
        self.widgetlogo.resize(1898,869)
        self.thread = Worker()
        self.main_to_thread.connect(self.thread.receive_from_main)
        self.thread.sig.connect(self.update_classification)
        self.thread.sig_face.connect(self.face_image)
        self.thread.sig_eye.connect(self.eye_image)
        self.thread.sig_pre.connect(self.sig_prefn)
        self.thread.sig_fps.connect(self.sig_fpsfn)
        self.thread.sig_eyeopen.connect(self.sig_eyeopenfn)
    def sig_eyeopenfn(self,open):
        self.labeyeopen.setText("{:.2f}".format(open))
    def sig_fpsfn(self,fps):
        self.labfps.setText("{:.1f}".format(fps))

    def genemessage(self):
        capmum = self.comboxcapture.currentIndex()
        eyewidth = self.sboxwidth.value()
        eyeheight = self.sboxheigth.value()
        eyethread = self.dsboxthread.value()
        eyefliternum = self.sboxeyenum.value()
        fliternum = self.sboxfilternum.value()
        return [capmum,eyewidth,eyeheight,eyethread,eyefliternum,fliternum]

    def setimage(self):
        """
        设置图片
        :return:
        """
        self.labeye.setScaledContents(True)
        self.labface.setScaledContents(True)
        self.setWindowTitle("智能眼动轮椅控制平台")
        self.setWindowIcon(QIcon(":/image/haligong.png"))
        logoimg = QImage(":/image/haligong.png")
        logoimg.scaled(250,250,Qt.KeepAspectRatio,Qt.SmoothTransformation)
        self.lablogo.setScaledContents(True)
        self.lablogo.setPixmap(QPixmap.fromImage(logoimg))
        self.btnentersys.setStyleSheet(
            'QPushButton{background-color: rgba(3, 60, 152,99);color: rgb(255, 255, 255);border-radius: 15px; border: 2px groove gray;border-style: outset;}'
            'QPushButton:hover{background-color: rgb(3, 60, 152)}')
        self.btnreturn.setStyleSheet(
            'QPushButton{background-color: rgba(3, 60, 152,99);color: rgb(255, 255, 255);border-radius: 10px; border: 2px groove gray;border-style: outset;}'
            'QPushButton:hover{background-color: rgb(3, 60, 152)}')
        self.btnstart.setStyleSheet(
            'QPushButton{background-color: rgba(3, 60, 152,99);color: rgb(255, 255, 255);border-radius: 10px; border: 2px groove gray;border-style: outset;}'
            'QPushButton:hover{background-color: rgb(3, 60, 152)}')
        self.labup.setScaledContents(True)
        self.labright.setScaledContents(True)
        self.lableft.setScaledContents(True)
        self.labrun.setScaledContents(True)
        self.labpreeye.setScaledContents(True)
        self.preleftimage = QImage(":image/preleft.png")
        self.prerightimage = QImage(":image/preright.png")
        self.preupimage = QImage(":image/preup.png")
        self.up_false =  QImage(":image/up.png")
        self.up_true =  QImage(":image/up_2.png")
        self.right_false =  QImage(":image/right.png")
        self.right_true =  QImage(":image/right_2.png")
        self.left_false =  QImage(":image/left.png")
        self.left_true =  QImage(":image/left_2.png")
        self.run =  QImage(":image/run.png")
        self.stop =  QImage(":image/stop.png")

    def showtime(self):
        time = QDateTime.currentDateTime().toString("yyyy/MM/dd HH:mm:ss")
        self.labtime.setText("{}".format(time))
        self.labtime_2.setText("{}".format(time))
        self.widgetlogo.receive()

    def buttonClicked(self):
        self.btnentersys.clicked.connect(lambda :self.setCurrentIndex(1))
        self.btnreturn.clicked.connect(self.btnreturnfn)
        self.btnstart.clicked.connect(self.btnstartfn)
        self.btnstartnum = 0
        self.comboxcapture.activated.connect(lambda :self.main_to_thread.emit(self.genemessage()))
        self.sboxwidth.valueChanged.connect(lambda :self.main_to_thread.emit(self.genemessage()))
        self.sboxheigth.valueChanged.connect(lambda :self.main_to_thread.emit(self.genemessage()))
        self.dsboxthread.valueChanged.connect(lambda :self.main_to_thread.emit(self.genemessage()))
        self.sboxeyenum.valueChanged.connect(lambda :self.main_to_thread.emit(self.genemessage()))
        self.sboxfilternum.valueChanged.connect(lambda :self.main_to_thread.emit(self.genemessage()))

    def btnreturnfn(self):
        self.setCurrentIndex(0)

    def btnstartfn(self):
        # if self.btnstartnum % 2 == 0:
        #     self.btnstart.setText("开始运行")
        #     self.btnstart.setStyleSheet('background-color: rgba(3, 60, 152,99);'
        #                                 'color: rgb(255, 255, 255);border-radius: 10px; border: 2px groove gray;border-style: outset;')
        #
        # else:
        #     self.btnstart.setText("结束运行")
        #     self.btnstart.setStyleSheet(
        #         'background-color: rgb(214, 0, 4);color: rgb(255, 255, 255);border-radius: 10px; border: 2px groove gray;border-style: outset;')
        self.main_to_thread.emit(self.genemessage())
        self.thread.start()

    def update_classification(self,text):
        if text == 1:
            self.labmoveorient.setText("前进")
            self.labright.setPixmap(QPixmap.fromImage(self.right_false))
            self.lableft.setPixmap(QPixmap.fromImage(self.left_false))
            self.labup.setPixmap(QPixmap.fromImage(self.up_true))
            self.labrun.setPixmap(QPixmap.fromImage(self.run))
        if text == 0:  #左
            self.labmoveorient.setText("左转")
            self.labright.setPixmap(QPixmap.fromImage(self.right_false))
            self.lableft.setPixmap(QPixmap.fromImage(self.left_true))
            self.labup.setPixmap(QPixmap.fromImage(self.up_false))
            self.labrun.setPixmap(QPixmap.fromImage(self.run))
        if text == 2:  #右
            self.labmoveorient.setText("右转")
            self.labright.setPixmap(QPixmap.fromImage(self.right_true))
            self.lableft.setPixmap(QPixmap.fromImage(self.left_false))
            self.labup.setPixmap(QPixmap.fromImage(self.up_false))
            self.labrun.setPixmap(QPixmap.fromImage(self.run))
        if text == 3:
            self.labmoveorient.setText("停止")
            self.labright.setPixmap(QPixmap.fromImage(self.right_false))
            self.lableft.setPixmap(QPixmap.fromImage(self.left_false))
            self.labup.setPixmap(QPixmap.fromImage(self.up_false))
            self.labrun.setPixmap(QPixmap.fromImage(self.stop))

    def sig_prefn(self,premess):
        if premess == 0:
            self.labpreorient.setText("前方")
            self.labpreeye.setPixmap(QPixmap.fromImage(self.preupimage))
        if premess == 1:
            self.labpreorient.setText("左方")
            self.labpreeye.setPixmap(QPixmap.fromImage(self.preleftimage))
        if premess == 2:
            self.labpreorient.setText("右方")
            self.labpreeye.setPixmap(QPixmap.fromImage(self.prerightimage))

    def eye_image(self,image_eye):
        image = cv2.cvtColor(image_eye, cv2.COLOR_BGR2RGB)
        height, width, bytesPerComponent = image.shape
        print("h:{},w:{}".format(height,width))
        bytesPerLine = bytesPerComponent * width
        q_image = QtGui.QImage(image.data, width, height, bytesPerLine,
                               QtGui.QImage.Format_RGB888).scaled(self.labeye.width(), self.labeye.height(),Qt.KeepAspectRatio,Qt.SmoothTransformation)
        self.labeye.setPixmap(QtGui.QPixmap.fromImage(q_image))

    def face_image(self,image):
        # print("debug",image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, bytesPerComponent = image.shape
        print("h:{},w:{}".format(height, width))
        bytesPerLine = bytesPerComponent * width
        q_image = QtGui.QImage(image.data, width, height, bytesPerLine,
                               QtGui.QImage.Format_RGB888).scaled(self.labface.width(), self.labface.height(),Qt.KeepAspectRatio,Qt.SmoothTransformation)
        self.labface.setPixmap(QtGui.QPixmap.fromImage(q_image))

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
    sig = pyqtSignal(int)
    sig_face = pyqtSignal(object)
    sig_eye = pyqtSignal(object)
    sig_pre = pyqtSignal(int)
    sig_fps = pyqtSignal(float)
    sig_eyeopen = pyqtSignal(float)
    def __init__(self,parent=None):
        super(Worker, self).__init__(parent)
        self.stop_flag = False
        self.RIGHT_EYE_START = 37 - 1
        self.RIGHT_EYE_END = 42 - 1
        self.LEFT_EYE_START = 43 - 1
        self.LEFT_EYE_END = 48 - 1
        self.use_cpu_dlib = True
        self.predictor_path = "./dlib/shape_predictor_68_face_landmarks.dat"
        self.predictor = dlib.shape_predictor(self.predictor_path)
        self.time_list = []
        self.settings = QSettings("config.ini", QSettings.IniFormat)
        self.serial_servo = Series()
        serialname = self.settings.value("SETUP/SERIAL_SERVO")
        self.serial_servo.setPortName(serialname)
        self.serial_servo.setBaudRate(QSerialPort.BaudRate.Baud115200)
        self.serial_servo.open(QSerialPort.WriteOnly)
        # if not self.serial_servo.isOpen():
        #     QMessageBox.warning(self, '警告', "舵机控制串口打开失败！")

    def receive_from_main(self,message):
        """
        接受主线程的信号
        :param message:
        :return:
        """
        self.cap_num,self.eye_w, self.eye_h,self.EYE_EAR,self.EYE_EAR_BEYOND_TIME,self.FLITER_NUM = message[0],message[1],message[2],message[3],message[4],message[5]
        #self.EYE_EAR_BEYOND_TIME 这个值设置的越小，眨眼所需的时间间隔越小
        #self.EYE_EAR  EAR阈值 眨眼判定的幅度越大，越容易判断

    def eye_aspect_ratio(self,eye):  # 计算EAR
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def run(self):
        self.cap = cv2.VideoCapture(self.cap_num)

        self.time_list = []
        blink_counter = False
        frame_counter = 0
        frame_counter_stop = 0
        blink_counter_stop = False

        a, w, dd = 0, 0, 0
        # serialFd = Series()
        # serialFd.

        # print("可用端口名>>>", serialFd.name)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print("device is ", device)
        model = GoogleNet3()
        model.load_state_dict(torch.load("./weight/Basic_Epoch_3_Accuracy_0.93.pth"))
        # 近中远
        model = model.to(device)
        transformer = create_transform()

        self.cap.set(3, 640)
        self.cap.set(4, 480)
        # print(f'camera width = {cap.get(3)}\ncamera height = {cap.get(4)}')
        offset_pixelY = -10
        offset_pixelX = 0

        if self.use_cpu_dlib == True:
            detector = dlib.get_frontal_face_detector()
        else:
            detector = dlib.cnn_face_detection_model_v1("./mmod_human_face_detector.dat")
        fps = 0.0
        read_flag,_ = self.cap.read()
        while read_flag:
            eye_w, eye_h = self.eye_w, self.eye_h
            pt_pos = []
            t1 = time.time()
            _, frame = self.cap.read()
            dets = detector(frame, 0)
            temp = len(dets)
            if temp == 0:
                self.serial_servo.write('r'.encode())
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
                    self.sig_eyeopen.emit(ear)
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
                            self.serial_servo.write('r'.encode())
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
                                        self.sig_eyeopen.emit(ear_stop)
                                        if ear_stop < self.EYE_EAR:  # 如果EAR小于阈值，开始累计连续眨眼次数
                                            frame_counter_stop += 1
                                        else:
                                            if frame_counter_stop >= self.EYE_EAR_BEYOND_TIME:  # 连续帧计数超过EYE_EAR_BEYOND_TIME的帧数时，累加超时次数（blink_counter+1）并给予提示警告
                                                blink_counter_stop = True
                                            frame_counter_stop = 0
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
                    if crop_eye.shape[0] > 0 and crop_eye.shape[1]>0:
                        print("crop_eye.shape",crop_eye.shape)
                        inputs = transformer(crop_eye).to(device)
                        outputs = model(inputs.unsqueeze(0))
                        _, y_pred = torch.max(outputs, dim=1)
                        y_pred = y_pred.item()
                        self.sig_pre.emit(y_pred)
                        if y_pred == 1:
                            w += 1
                            if w % self.FLITER_NUM == 0:
                                self.sig.emit(int(y_pred))
                                self.serial_servo.write('r'.encode())
                                self.serial_servo.write('w'.encode())
                        if y_pred == 0:
                            a += 1
                            if a % self.FLITER_NUM == 0:
                                self.sig.emit(int(y_pred))
                                self.serial_servo.write('r'.encode())
                                self.serial_servo.write('a'.encode())
                        if y_pred == 2:
                            dd += 1
                            if dd % self.FLITER_NUM == 0:
                                self.sig.emit(int(y_pred))
                                self.serial_servo.write('r'.encode())
                                self.serial_servo.write('d'.encode())
                        fps = (fps + (1. / (time.time() - t1))) / 2
                        self.sig_fps.emit(fps)
                        frame = cv2.putText(frame, f"fps={fps:5.1f}", (0, 100),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    else:
                        continue
            self.sig_face.emit(frame)
        self.cap.release()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = mainWindow()
    ui.show()
    sys.exit(app.exec_())
