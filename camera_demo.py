import torch
import torch.nn as nn
import serial.tools.list_ports
import os
import cv2
from models import create_dataloader,Classification,CnnLstm
from option import BasciOption
import dlib
import numpy as np
import time
from models import GoogleNet,GoogleNet3
from models import create_transform

class VideoPreprocess(nn.Module):

    def __init__(self, opt):
        super(VideoPreprocess, self).__init__()
        self.opt = opt
        self.is_success = True

        # self.end_frame = 10
        self.predictor_path = "./dlib/shape_predictor_68_face_landmarks.dat"
        self.video_size = (200, 50)  # w h
        self.offset_up_pixelY = -10
        self.offset_bottom_pixelY = 5
        self.offset_pixelX = 0
        self.offset_pixelX = 0
        print(f'self opt = {self.opt}')


    def video_process(self):



        print(videoCapture.isOpened())
        count=0
        while self.is_success:
            self.is_success, frame = videoCapture.read()  # opencv 读取图像的格式为 H,W,C https://blog.csdn.net/sinat_28704977/article/details/89969199
            # count+=1

            # print("count",count-self.start_frame)
            # print("count",count)

            # 参数详解：https://blog.csdn.net/weixin_44493841/article/details/93488126
            # detector的第二个参数： 0 ：代表检测原始图像大小，1：代表将原始图像放大一倍后检测，有助于检测小人脸
            dets = detector(frame, 0)

            pt_pos = []
            # eye_w, eye_h = 100, 50
            eye_w, eye_h = 40, 30
            # print('dets---',dets[0])
            if dets: # 判断是否检测到了人脸，如果检测到两个人脸的话 则会len(dets) = 2
                count += 1
                for k, d in enumerate(dets):

                    # print('count----:',count)
                    # print("dets{}".format(d))
                    # print(
                    #     "Detection{}: Left: {} Top: {} Right: {} Bottom: {}".format(
                    #         k, d.left(), d.top(), d.right(), d.bottom()))
                    """
                    TODO：
                    我们程序中只需要检测一个人脸，所以需要通过检测框的面积大小来将非坐在轮椅上的人过滤掉，保留面积最大对应的dets
                    下面的操作都只需要对一个人的检测即可，感觉可以放在for循环外面做
                    """
                    shape = predictor(frame, d)# 预测人脸形状大小
                    for index, pt in enumerate(shape.parts()):
                        pt_pos.append((pt.x, pt.y))#人脸坐标点

                    # cv2.waitKey(30)

                    left_eye = frame[
                               pt_pos[37][1] + self.offset_up_pixelY:pt_pos[37][1]
                                                                 + eye_h + self.offset_bottom_pixelY,
                               pt_pos[36][0] + self.offset_up_pixelY:pt_pos[36][
                                                                     0] + eye_w +self.offset_bottom_pixelY]

                    right_eye = frame[
                                pt_pos[44][1] +self.offset_up_pixelY:pt_pos[44][1]
                                                                  + eye_h + self.offset_bottom_pixelY,
                                pt_pos[42][0] +self.offset_up_pixelY:pt_pos[42][
                                                                      0] + eye_w +self.offset_bottom_pixelY]
                    # print("left",left_eye.shape)
                    # print("right_eye.shape",right_eye.shape)
                    if self.opt.save_img == True:
                        # if not os.path.exists(
                        #         os.path.join(opt.save_path, 'left')):
                        #     os.makedirs(os.path.join(opt.save_path, 'left'))
                        #
                        # if not os.path.exists(
                        #         os.path.join(opt.save_path, 'right')):
                        #     os.makedirs(os.path.join(opt.save_path, 'right'))

                        if not os.path.exists(
                                os.path.join(self.opt.save_path, 'concat')):
                            os.makedirs(os.path.join(self.opt.save_path, 'concat'))

                        crop_eye = np.concatenate((left_eye, right_eye),
                                                  axis=1)
                        # if left_eye.shape[:2] == (50, 100) and right_eye.shape[:2] == (50, 100):  # 格式为 H*W*C
                        # if left_eye.shape[:2] == (eye_h, eye_w) and right_eye.shape[:2] == (eye_h, eye_w):  # 格式为 H*W*C
                        crop_eye = cv2.resize(crop_eye, self.video_size)
                        # cv2.imwrite(os.path.join(self.opt.save_path, 'concat', f'{count}.jpg'), crop_eye)



                        cv2.imshow('crop_eyes', crop_eye)
                        cv2.imshow('origin_pic', frame)



            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        videoCapture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
	plist = list(serial.tools.list_ports.comports())
	if len(plist) <= 0:
        print("没有发现端口!")
    else:
        plist_0 = list(plist[0])
        serialName = plist_0[0]
        serialFd = serial.Serial(serialName, 115200, timeout=60)
        print("可用端口名>>>", serialFd.name)
    opt = BasciOption(train_flag=False).initialize()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GoogleNet3()

    # model.load_state_dict(torch.load(r"E:\Gaze_Project\Intelligent_Wheelchair\output\train\weights\exp_28\Basic_Epoch_5_Accuracy_0.93.pth"))
    #左中右
    model.load_state_dict(torch.load(r"./Basic_Epoch_3_Accuracy_0.93.pth"))
    #近中远
    model = model.to(device)
    transformer = create_transform(opt)
    #-------------------------------------#
    #   调用摄像头
    #   capture=cv2.VideoCapture("1.mp4")
    #-------------------------------------#
    predictor_path = "./dlib/shape_predictor_68_face_landmarks.dat"
    video_size = (200, 50)  # w h
    # offset_up_pixelY = -10
    # offset_bottom_pixelY = 5
    # offset_left_pixelX = 0
    # offset_right_pixelX = 0
    offset_pixelY = -14
    offset_pixelX = 0

    predictor = dlib.shape_predictor(predictor_path)

    # 初始化dlib人脸检测器
    detector =dlib.cnn_face_detection_model_v1('./mmod_human_face_detector.dat')

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(1)

    cap.set(3,1280)
    cap.set(4,720)
    # videoCapture = cv2.VideoCapture(url)

    print(f'camera width = {cap.get(3)}\ncamera height = {cap.get(4)}')
    fps = 0.0

    while(True):
        t1 = time.time()
        # 读取某一帧
        ref,frame=cap.read()

        dets = detector(frame, 0)

        pt_pos = []
        # eye_w, eye_h = 100, 50
        eye_w, eye_h = 40, 30
        # print('dets---',dets[0])
        if dets:  # 判断是否检测到了人脸，如果检测到两个人脸的话 则会len(dets) = 2
            # count += 1
            for k, d in enumerate(dets):

                # print('count----:',count)
                # print("dets{}".format(d))
                # print(
                #     "Detection{}: Left: {} Top: {} Right: {} Bottom: {}".format(
                #         k, d.left(), d.top(), d.right(), d.bottom()))
                """
                TODO：
                我们程序中只需要检测一个人脸，所以需要通过检测框的面积大小来将非坐在轮椅上的人过滤掉，保留面积最大对应的dets
                下面的操作都只需要对一个人的检测即可，感觉可以放在for循环外面做
                """
                shape = predictor(frame, d)  # 预测人脸形状大小
                for index, pt in enumerate(shape.parts()):
                    pt_pos.append((pt.x, pt.y))  # 人脸坐标点

                # cv2.waitKey(30)

                # left_eye = frame[
                #            pt_pos[37][1] + offset_up_pixelY:pt_pos[37][1]
                #                                                  + eye_h + offset_bottom_pixelY,
                #            pt_pos[36][0] + offset_left_pixelX:pt_pos[36][
                #                                                      0] + eye_w + offset_right_pixelX]
                #
                # right_eye = frame[
                #             pt_pos[44][1] + offset_up_pixelY:pt_pos[44][1]
                #                                                   + eye_h + offset_bottom_pixelY,
                #             pt_pos[42][0] + offset_left_pixelX:pt_pos[42][
                #                                                       0] + eye_w + offset_right_pixelX]
                left_eye = frame[
                           pt_pos[37][1] + offset_pixelY:pt_pos[37][1]
                                                              + eye_h + offset_pixelY,
                           pt_pos[36][0] + offset_pixelX:pt_pos[36][
                                                                  0] + eye_w + offset_pixelX]

                right_eye = frame[
                            pt_pos[44][1] + offset_pixelY:pt_pos[44][1]
                                                               + eye_h + offset_pixelY,
                            pt_pos[42][0] + offset_pixelX:pt_pos[42][
                                                                   0] + eye_w + offset_pixelX]

                crop_eye = np.concatenate((left_eye, right_eye),
                                          axis=1)
                # if left_eye.shape[:2] == (50, 100) and right_eye.shape[:2] == (50, 100):  # 格式为 H*W*C
                # if left_eye.shape[:2] == (eye_h, eye_w) and right_eye.shape[:2] == (eye_h, eye_w):  # 格式为 H*W*C
                # crop_eye = cv2.resize(crop_eye, video_size)
                # cv2.imwrite(os.path.join(self.opt.save_path, 'concat', f'{count}.jpg'), crop_eye)

                cv2.imshow('crop_eyes', crop_eye)
                # crop_eye = cv2.cvtColor(crop_eye, cv2.COLOR_BGR2RGB)
                inputs = transformer(crop_eye).to(device)

                outputs = model(inputs.unsqueeze(0))
                _,y_pred = torch.max(outputs,dim = 1)
				print(y_pred)
                if y_pred == 1:
                    serialFd.write('w'.encode())
                elif y_pred == 0:
                	serialFd.write('a'.encode())
			    elif y_pred == 2:
					serialFd.write(“d”.encode())

                frame = cv2.putText(frame, "label= %.2f" % (y_pred), (0, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        fps = (fps + (1. / (time.time() - t1))) / 2
        frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('origin_pic', frame)


        # fps  = ( fps + (1./(time.time()-t1)) ) / 2
        # print("fps= %.2f"%(fps))
        # frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # cv2.imshow("video",frame)

        c= cv2.waitKey(1) & 0xff
        if c==27:
            cap.release()
            break
