# -*-coding:utf-8-*-
# date:2020-04-11
# author: Eric.Lee
# function : classify

import os
import torch
import cv2
import numpy as np
import json
import PIL
import torch
import torch.nn as nn

import numpy as np


import os

from datetime import datetime
import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet, model

#
class classify_imagenet_model(object):
    def __init__(
        self,
        model_path="model_best.pth.tar",
        img_size=112,
        num_classes=14,
    ):

        f = open("imagenet_msg.json", encoding="utf-8")  # 读取 json文件
        dict_ = json.load(f)
        f.close()
        self.classify_dict = dict_
        # print("-------------->>\n dict_ : \n",dict_)
        #
        print("classify model loading : ", model_path)
        # print('use model : %s'%(model_arch))
        model_ = EfficientNet.from_name(
            "efficientnet-b0", num_classes=num_classes, image_size=img_size
        )

        use_cuda = torch.cuda.is_available()

        device = torch.device("cuda:0" if use_cuda else "cpu")
        model_ = model_.to(device)
        model_.eval()  # 设置为前向推断模式

        # print(model_)# 打印模型结构

        # 加载测试模型
        if os.access(model_path, os.F_OK):  # checkpoint
            chkpt = torch.load(model_path, map_location=device)
            start_epoch = chkpt["epoch"]
            arch = chkpt["arch"]
            best_acc1 = chkpt["best_acc1"]
            model_.load_state_dict(chkpt["state_dict"])

            # print('load classify model : {}'.format(model_path))
        self.model_ = model_
        self.use_cuda = use_cuda
        self.img_size = img_size

    def predict(self, img, vis=False):  # img is align img
        with torch.no_grad():

            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            val_transforms = transforms.Compose(
                [
                    transforms.Resize(self.img_size, interpolation=PIL.Image.BICUBIC),
                    # transforms.CenterCrop(self.img_size),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
            img_ = val_transforms(img)
            img_ = img_.unsqueeze_(0)

            if self.use_cuda:
                img_ = img_.cuda()  # (bs, 3, h, w)

            pre_ = self.model_(img_)
            print(pre_)
            outputs = F.softmax(pre_, dim=1)
            outputs = outputs[0]

            output = outputs.cpu().detach().numpy()
            output = np.array(output)

            max_index = np.argmax(output)

            score_ = output[max_index]
            # print("max_index:",max_index)
            # print("name:",self.label_dict[max_index])
            return max_index, self.classify_dict[str(max_index)], score_


if __name__ == "__main__":
    model = classify_imagenet_model()
    path = "test.jpg"
    img = PIL.Image.open(path)
    index_id, label, score = model.predict(img)
    print(index_id, label, score)
