from PyQt5.QtCore import QTimer,QPointF
from PyQt5.QtGui import QColor, QBrush,QFont,QPainter,QPixmap,QPen
from PyQt5.QtWidgets import QWidget
import math
import random

"""
需要传入角度 和 正反转 是否转动 
"""

class Drawlabel(QWidget):
    def __init__(self,form):
        super(Drawlabel, self).__init__()
        self.resize(1002, 491)  #设置大小

        #color
        self.pointcolor = QColor(91, 153, 74)
        self.textsize = 24
        self.linewidth = 2
        self.linecolor = QColor(241, 225, 100)
        self.textcolor =QColor(0, 0, 0)

    def receive(self):
        self.update()

    #绘画事件
    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        painter.setRenderHint(QPainter.Antialiasing)

        self.drawpixmap(event, painter)
        painter.end()

    def drawpixmap(self, event, painter):
        pixmap_ = QPixmap(":/image/haligong.png")
        size = self.size()
        randomx = random.randint(0, size.width()-size.width() // 5)
        randomy = random.randint(0, size.height()-size.width() // 5)
        painter.drawPixmap(randomx, randomy, size.width() // 5, size.width() // 5, pixmap_)
