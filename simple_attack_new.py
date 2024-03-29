# Helper function for extracting features from pre-trained models
import sys, os
import argparse
import torch
import torch.nn as nn
import cv2
import numpy as np
import glob

from attacker import Attacker
from loader import ImageNet_A
from models.models import model_selection
from utils.Resnet import resnet152_denoise, resnet101_denoise, resnet152
from utils.Normalize import Normalize, Permute


class Ensemble(nn.Module):
    def __init__(self, model1, model2=None, model3=None):
        super(Ensemble, self).__init__()
        self.model1 = model1
        self.model2 = model2
        # self.model3 = model3

    def forward(self, x):
        logits1 = self.model1(x)
        logits2 = self.model2(x)
        # logits3 = self.model3(x)

        # fuse logits
        logits_e = (logits1 + logits2) / 2

        return logits_e


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='/raid/chenby/tianchi/imagenet/', type=str, help='path to data')
    parser.add_argument('--output_dir', default='./results/04_ensemble_nt_b5adv_200_500_step100_8/', type=str, help='path to results')
    parser.add_argument('--batch_size', default=8, type=int, help='mini-batch size')
    parser.add_argument('--steps', default=100, type=int, help='iteration steps')
    parser.add_argument('--max_norm', default=8, type=float, help='Linf limit')
    parser.add_argument('--div_prob', default=0.9, type=float, help='probability of diversity')
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, 'images')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    model1 = model_selection(model_name='efficientnet-b5', advprop=True)  # efficientnet-b5
    model2 = model_selection(model_name='resnet50')  # efficientnet-b5
    model = Ensemble(model1, model2)
    model = nn.Sequential(
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        model
    )

    model.cuda()
    model.eval()

    # set dataset
    dataset = ImageNet_A(args.input_dir)
    loader = torch.utils.data.DataLoader(dataset, 
                                         batch_size=args.batch_size, 
                                         shuffle=False)

    # set attacker
    attacker = Attacker(steps=args.steps, 
                        max_norm=args.max_norm/255.0,
                        div_prob=args.div_prob,
                        device=torch.device('cuda'),
                        low=200,
                        high=500)

    for ind, (img, label_true, label_target, filenames) in enumerate(loader):

        # run attack
        adv = attacker.attack(model, img.cuda(), label_true.cuda(), label_target.cuda())

        # save results
        for bind, filename in enumerate(filenames):
            out_img = adv[bind].detach().cpu().numpy()
            delta_img = np.abs(out_img - img[bind].numpy()) * 255.0

            print('Attack on {}:'.format(os.path.split(filename)[-1]))
            print('Max: {0:.0f}, Mean: {1:.2f}'.format(np.max(delta_img), np.mean(delta_img)))

            out_img = np.transpose(out_img, axes=[1, 2, 0]) * 255.0
            out_img = out_img[:, :, ::-1]

            out_filename = os.path.join(output_dir, os.path.split(filename)[-1])
            cv2.imwrite(out_filename, out_img)