# Copyright (C) 2018 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the MIT License. To view a copy of this license,
# visit https://opensource.org/licenses/MIT.

"""Entry point for testing AttGAN network."""

import argparse
import json
import os
from os.path import join
import cv2

import torch
import torch.utils.data as data
import torchvision.utils as vutils

from attgan import AttGAN
from data import check_attribute_conflict
from helpers import Progressbar
from utils import find_model

import torchvision.transforms as transforms
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt



def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', dest='experiment_name', default='384_shortcut1_inject1_none_hq')
    parser.add_argument('--test_int', dest='test_int', type=float, default=1.0)
    parser.add_argument('--num_test', dest='num_test', type=int)
    parser.add_argument('--load_epoch', dest='load_epoch', type=str, default='latest')
    parser.add_argument('--custom_img', action='store_true')
    parser.add_argument('--custom_data', type=str, default='./data/custom')
    parser.add_argument('--custom_attr', type=str, default='./data/list_attr_custom.txt')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--image', dest='image', type=str, default='test2.png')
    parser.add_argument('--attr', dest='attrnum', type=int, default=7)
    return parser.parse_args(args)

args_ = parse()

print("---------------args_********************************")
print(args_)

with open(join('output', args_.experiment_name, 'setting.txt'), 'r') as f:
    args = json.load(f, object_hook=lambda d: argparse.Namespace(**d))

args.test_int = args_.test_int
args.num_test = args_.num_test
args.gpu = args_.gpu
args.load_epoch = args_.load_epoch
args.multi_gpu = args_.multi_gpu
args.custom_img = True
args.custom_data = args_.custom_data
args.custom_attr = args_.custom_attr
args.n_attrs = len(args.attrs)
args.betas = (args.beta1, args.beta2)
args.image = args_.image
args.attrnum = args_.attrnum

print("***************args******************************")
print(args)


if args.custom_img:
    output_path = join('output', args.experiment_name, 'custom_testing')
    from data import Custom
    test_dataset = Custom(args.custom_data, args.custom_attr, args.img_size, args.attrs)
else:
    output_path = join('output', args.experiment_name, 'sample_testing')
    if args.data == 'CelebA':
        from data import CelebA
        test_dataset = CelebA(args.data_path, args.attr_path, args.img_size, 'test', args.attrs)
    if args.data == 'CelebA-HQ':
        from data import CelebA_HQ
        test_dataset = CelebA_HQ(args.data_path, args.attr_path, args.image_list_path, args.img_size, 'test', args.attrs)
os.makedirs(output_path, exist_ok=True)


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread(args.image)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
height, width = img.shape[0], img.shape[1]
print(height, width)


if len(faces) > 0:
    (x, y, w, h) = faces[0]    

    bottom_x = max(int(x - w/2), 1)
    bottom_y = max(int(y - h/2), 1)
    cropSize = max(w, h)
    
    top_x = min(width, int(bottom_x+2*cropSize))
    top_y = min(height, int(bottom_y+2*cropSize))

    cropSize = int(min(top_x-bottom_x, top_y-bottom_y)/2)
    cropFace =  img[bottom_y:bottom_y+2*cropSize, bottom_x:bottom_x+2*cropSize]
    
    # cv2.imshow("ss",cropFace)
    # cv2.waitKey()  
    # exit 

    tf = transforms.Compose([
                transforms.Resize(args.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
   
    attgan = AttGAN(args)
    attgan.load(find_model(join('output', args.experiment_name, 'checkpoint'), args.load_epoch))
    progressbar = Progressbar()
    attgan.eval()
    cropFace = cv2.cvtColor(cropFace, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cropFace)

    img_a = tf(pil_image)       
    img_a = torch.unsqueeze(img_a, dim=0)
    att_a = torch.tensor( (np.array([[-1,1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1]]) + 1) // 2)


    img_a = img_a.cuda() if args.gpu else img_a
    att_a = att_a.cuda() if args.gpu else att_a
    att_a = att_a.type(torch.float)

    
    att_b_list = [att_a]


    for i in range(args.n_attrs):
        tmp = att_a.clone()
        tmp[:, i] = 1 - tmp[:, i]
        tmp = check_attribute_conflict(tmp, args.attrs[i], args.attrs)
        att_b_list.append(tmp)

    with torch.no_grad():
        samples = [img_a]
        for i, att_b in enumerate(att_b_list):
            att_b_ = (att_b * 2 - 1) * args.thres_int
            if i > 0:
                att_b_[..., i - 1] = att_b_[..., i - 1] * args.test_int / args.thres_int
            samples.append(attgan.G(img_a, att_b_))
        samples = torch.cat(samples, dim=3)

        out_file = "result.jpg"
      
        vutils.save_image(
            samples, join(output_path, out_file),
            nrow=1, normalize=True, range=(-1., 1.)
        )


        newimg = cv2.imread(join(output_path, out_file))    

        h, w = newimg.shape[0], newimg.shape[1]
        ind = min(args.attrnum, 13)
        ind = max(ind, 1)
        cropnewimg = newimg[0:h, h*(ind+1):h*(ind+2)]

        resized = cv2.resize(cropnewimg, (2*cropSize, 2*cropSize), interpolation = cv2.INTER_AREA)

        
        

        img[bottom_y:bottom_y+2*cropSize, bottom_x:bottom_x+2*cropSize] = resized

        cv2.imwrite("result.jpg", img)
        cv2.imshow("sss", img)
        cv2.waitKey()  
        print('{:s} done!'.format(out_file))

