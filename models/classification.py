import cv2
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


class Classification(torch.nn.Module):

    def __init__(self):
        super(Classification, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=3, padding=0)
        # lstm


        # self.conv2 = nn.Conv2d(20, 50, kernel_size=3)
        self.fc1 = nn.Linear(47520, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))   # [N, 20, 24, 99]  N*C*H*W
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # print("Debug", x.shape)  # [N, 20, 24, 99]  N*C*H*W
        x = x.view(-1, 47520)  # 全连接之前需要改变下维度， 变成(N,C*H*W) ,由于batch不是每个循环都一样多，所以使用x.view
        x = F.relu(self.fc1(x))
        x = self.fc2(x)    # [1, 2]
        # print("Debug", x.shape) [47520]
        return x


class CnnLstm(torch.nn.Module):

    def __init__(self):
        super(CnnLstm, self).__init__() #3*50*200
        self.seq = 20
        self.hidden_size = 256
        self.conv1 = nn.Conv2d(3, 20, kernel_size=3, padding=0)
        # 将变量展开成一维，或者用一个fc层
        self.fc1 = nn.Linear(47520, 512)
        # lstm
        self.lstm = nn.LSTM(input_size=512, hidden_size=self.hidden_size, num_layers=1, batch_first=True)

        # self.fc2 = nn.Linear(256, 2)  #[N,hidden,classes]
        # self.fc2 = nn.Linear(2560, 2)  #[N,hidden,classes]
        self.fc2 = nn.Linear(self.seq * self.hidden_size, 1)  #[N,hidden,classes]

    def forward(self, x_frame_concat):
        cnn_embed_seq = []
        #print('debug input size :', x_frame_concat.size())  #[2, 10, 3, 50, 200] 当label.txt中只有一行的时候输出为 [10, 3, 50, 200]，后续就不太对
        assert x_frame_concat.size(1) == self.seq, 'frame is not expect'
        for seq in range(x_frame_concat.size(1)):
            #print(f'seq:{seq}')
            #print(x_frame_concat[:,seq, :, :, :],x_frame_concat.size())
            # x = F.relu(F.max_pool2d(self.conv1(x[seq, :, :, :]), (2, 2)))  # [N, 20, 24, 99]  N*C*H*W
            x = F.relu(F.max_pool2d(self.conv1(x_frame_concat[:,seq, :, :, :]), (2, 2)))  # [N, 20, 24, 99]  N*C*H*W
            #print(f'conv1.shape: {x.shape}')
            x = x.view(x.size(0), -1)   # [N, 20*24*99= 47520]
            #print(f'x.view.shape: {x.shape}')
            x = F.relu(self.fc1(x))     # [N, 512]
            #print(f'fc1(x).shape: {x.shape}')
            cnn_embed_seq.append(x)
        #print(f'cnn_embed_seq.shape: {len(cnn_embed_seq)}')
        # cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0)
        # print()
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).permute(1, 0, 2)  # [N ,time_seq, 512]  [2,512,10]
        # cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0)  # [N ,time_seq, 512]
        #print(f'cnn_embed_seq_T.shape: {cnn_embed_seq.size()}')
        x, (h_n,c_n) = self.lstm(cnn_embed_seq)
        #print(f'lstm.shape: {len(x)}')
        # print(f'lstm.shape: {x.size()}')
        # x = F.relu(self.fc2(x))

        # x = self.fc2(x.reshape([2, 2560]))
        # x = F.sigmoid(self.fc2(x.reshape([2,-1])))
        x = F.sigmoid(self.fc2(x.reshape((-1,self.seq * 256))))

        #print(f'fc2(x).shape: {x.size()}')
        return x

class Inception(nn.Module):
    def __init__(self,in_channels):
        super(Inception,self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels,16,kernel_size=1)
        self.branch5x5_1 = nn.Conv2d(in_channels,16,kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16,24,kernel_size=5,padding=2)
        self.branch3x3_1 = nn.Conv2d(in_channels,16,kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16,24,kernel_size=3,padding=1)
        self.branch3x3_3 = nn.Conv2d(24,24,kernel_size=3,padding=1)
        self.branch_pool = nn.Conv2d(in_channels,24,kernel_size=1)

    def forward(self,x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)
        branch_pool = F.avg_pool2d(x,kernel_size=3,stride=1,padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch_pool,branch1x1,branch3x3,branch5x5,]
        concat = torch.cat(outputs,dim=1)
        return concat
#3*50*200
class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet,self).__init__()
        self.conv1 = nn.Conv2d(3,10,kernel_size=5,padding=2)
        self.incep = Inception(in_channels=10)
        self.conv2 = nn.Conv2d(88,32,kernel_size=5,padding=0)
        self.conv3 = nn.Conv2d(32,16,kernel_size=3,padding=0)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(1472,1)
    def forward(self,x):
        # batch=x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = F.relu(self.mp(self.conv3(x)))
        x = x.view(1,-1)
        # print(x.size())
        x = F.sigmoid(self.fc(x))
        return x

class ResidualBlock(nn.Module):
    def __init__(self,channels):
        super(ResidualBlock,self).__init__()
        self.conv1 = nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(channels,channels,kernel_size=3,padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x+y)

class GoogleNet3(nn.Module):
    def __init__(self):
        super(GoogleNet3,self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5, padding=2)
        self.incep = Inception(in_channels=10)
        self.conv2 = nn.Conv2d(88, 32, kernel_size=5, padding=0)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=0)
        self.mp = nn.MaxPool2d(2)
        self.resnet = ResidualBlock(32,)
        self.fc = nn.Linear(1472, 3)
    def forward(self,x):
        batch=x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.resnet(x)
        x = F.relu(self.mp(self.conv3(x)))
        x = x.view(batch,-1)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.sigmoid(self.fc3(x))
        x = self.fc(x)
        return x


# if __name__ == '__main__':
#     inputs = torch.randn((1,20,3,50,200))
#     print(inputs.shape)
#     model = CnnLstm()
#     model.train()
#     output = model(inputs)
#     print(f'output={output}')
