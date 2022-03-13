# -*- coding: utf-8 -*-

"""
Module implementing mainWindow.
"""
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot,QTimer,QDateTime,Qt
from PyQt5.QtWidgets import QMainWindow,QStackedWidget
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
from PyQt5.QtCore import QThread, pyqtSignal
from imutils import face_utils
from scipy.spatial import distance
import time
from drawlabel import Drawlabel

class mainWindow(QStackedWidget, Ui_StackedWidget):
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
        self.widgetlogo.resize(1002,444)
    def setimage(self):
        """
        设置图片
        :return:
        """
        self.setWindowTitle("智能眼动轮椅控制平台")
        self.setWindowIcon(QIcon(":/image/haligong.png"))
        logoimg = QImage(":/image/haligong.png")
        logoimg.scaled(250,250,Qt.KeepAspectRatio,Qt.SmoothTransformation)
        self.lablogo.setScaledContents(True)
        self.lablogo.setPixmap(QPixmap.fromImage(logoimg))

    def showtime(self):
        time = QDateTime.currentDateTime().toString("yyyy/MM/dd HH:mm:ss")
        self.labtime.setText("{}".format(time))
        self.labtime_2.setText("{}".format(time))
        self.widgetlogo.receive()

    def buttonClicked(self):
        self.btnentersys.clicked.connect(lambda :self.setCurrentIndex(1))
        self.btnreturn.clicked.connect(self.btnreturnfn)
    def btnreturnfn(self):
        self.setCurrentIndex(0)
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = mainWindow()
    ui.show()
    sys.exit(app.exec_())
