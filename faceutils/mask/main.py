#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os.path as osp
import time
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms

from .model import BiSeNet
#from model import BiSeNet
from fvcore.nn import FlopCountAnalysis

class FaceParser:
    def __init__(self, device="cpu"):
        #mapper = [0, 1, 2, 3, 4, 5, 0, 11, 12, 0, 6, 8, 7, 9, 13, 0, 0, 10, 0]
        mapper = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        self.device = device
        self.dic = torch.tensor(mapper, device=device).unsqueeze(1)
        save_pth = osp.split(osp.realpath(__file__))[0] + '/resnet.pth'

        net = BiSeNet(n_classes=19)
        net.load_state_dict(torch.load(save_pth, map_location=device))
        self.net = net.to(device).eval()
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


    def parse(self, image: Image):
        assert image.shape[:2] == (512, 512)
        with torch.no_grad():
            image = self.to_tensor(image).to(self.device)
            image = torch.unsqueeze(image, 0)
            out = self.net(image)[0]
            parsing = out.squeeze(0).argmax(0)
        parsing = torch.nn.functional.embedding(parsing, self.dic)
        return parsing.float().squeeze(2)

if __name__ == "__main__":
    '''
    mapper = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

    faceparser = FaceParser(device="cpu")
    x = torch.randn(16, 3, 256, 256)[0]
    parsing = x.squeeze(0).argmax(0)
    print('x', x.shape)              # x torch.Size([3, 256, 256])
    print('parsing', parsing.shape)  # parsing torch.Size([256, 256])
    
    dic = torch.tensor(mapper, device="cpu").unsqueeze(1)  # dic torch.Size([19, 1])
    parsing = torch.nn.functional.embedding(parsing, dic)  # parsing_after torch.Size([256, 256, 1])
    parsing = parsing.float().squeeze(2)                   # parsing_after_squeeze torch.Size([256, 256])
    '''
    #start_time = time.time()
    faceparser = FaceParser(device="cuda:3")
    x = torch.randn(1, 3, 256, 256).to("cuda:3")
    start_time = time.time()
    for _ in range(1000):
        feat8, feat16, feat32 = faceparser.net(x)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    #flops = FlopCountAnalysis(faceparser.net, x)
    #print(f"FLOPs: {flops.total()}")  #
    feat8, feat16, feat32 = faceparser.net(x)
    print('feat8', feat8.shape)    # feat8 torch.Size([16, 19, 256, 256])
    print('feat16', feat16.shape)  # feat16 torch.Size([16, 19, 256, 256])
    print('feat32', feat32.shape)  # feat32 torch.Size([16, 19, 256, 256])
    #'''