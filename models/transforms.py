'''
Author: your name
Date: 2021-01-25 14:27:44
LastEditTime: 2021-01-25 15:10:26
LastEditors: your name
Description: In User Settings Edit
FilePath: \modify_module\models\transforms.py
'''

from torchvision import transforms

def create_transform():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((200, 50)),
        #transforms.Resize((50, 50)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485,], std=[0.229])])
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transform
