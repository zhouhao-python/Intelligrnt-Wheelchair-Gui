from PyQt5.QtSerialPort import QSerialPort

class Series(QSerialPort):
    def __init__(self):
        super(Series,self).__init__()
        self.setBaudRate(QSerialPort.BaudRate.Baud19200)
        self.setDataBits(QSerialPort.DataBits.Data8)
        self.setParity(QSerialPort.Parity.NoParity)
        self.setStopBits(QSerialPort.StopBits.OneStop)
        self.setFlowControl(QSerialPort.FlowControl.NoFlowControl)



