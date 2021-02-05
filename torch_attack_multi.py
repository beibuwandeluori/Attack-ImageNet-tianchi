# Helper function for extracting features from pre-trained models
import sys, os
import argparse
import torch
import torch.nn as nn
import cv2
import numpy as np
import glob

from attacker import Attacker, AttackerPGD, AttackerTPGD, AttackerMIFGSM
import torchattacks
from loader import ImageNet_A, input_diversity
from models.models import model_selection
from utils.Resnet import resnet152_denoise, resnet101_denoise, resnet152
from utils.Normalize import Normalize, Permute, Resize


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
    # /raid/chenby/tianchi/imagenet/
    parser.add_argument('--input_dir', default='/raid/chenby/tianchi/imagenet/', type=str, help='path to data')
    parser.add_argument('--output_dir', default='./results/07_ensemble_MIM_step100_8_PGD_step50_4/', type=str, help='path to results')
    parser.add_argument('--batch_size', default=8, type=int, help='mini-batch size')
    parser.add_argument('--steps', default=100, type=int, help='iteration steps')
    parser.add_argument('--max_norm', default=8, type=float, help='Linf limit')
    parser.add_argument('--div_prob', default=0.9, type=float, help='probability of diversity')
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, 'images')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    # ensemble model
    model1 = model_selection(model_name='efficientnet-b5', advprop=False)  # efficientnet-b5
    model1 = nn.Sequential(
        Resize(input_size=[456, 456]),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        model1
    )
    model2 = model_selection(model_name='resnet50')  # efficientnet-b5
    model2 = nn.Sequential(
        Resize(input_size=[224, 224]),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        model2
    )
    model = Ensemble(model1, model2)
    # print(model)

    model.cuda()
    model.eval()

    # set dataset
    dataset = ImageNet_A(args.input_dir)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False)

    # set attacker
    # attacker = Attacker(steps=args.steps,
    #                     max_norm=args.max_norm/255.0,
    #                     div_prob=args.div_prob,
    #                     device=torch.device('cuda'),
    #                     low=200,
    #                     high=500)
    # attacker = torchattacks.PGD(model, eps=args.max_norm/255.0, alpha=2/255.0, steps=args.steps)

    # attacker = AttackerTPGD(model, eps=args.max_norm/255.0, alpha=2/255.0, steps=args.steps, low=200, high=500,
    #                        div_prob=args.div_prob)
    attacker = AttackerMIFGSM(model, eps=args.max_norm / 255.0, decay=1.0, steps=args.steps, low=200, high=500,
                              div_prob=args.div_prob)
    attacker_02 = AttackerPGD(model, eps=args.max_norm/2.0/255.0, alpha=2/255.0, steps=args.steps//2, low=200, high=500,
                           div_prob=args.div_prob)

    for ind, (img, label_true, label_target, filenames) in enumerate(loader):
        # flag = False
        # for filename in filenames:
        #     if '2776.jpg' in filename:
        #         print(filenames)
        #         print(os.path.join(output_dir, os.path.split(filenames[-1])[-1]))
        #         print(os.path.exists(os.path.join(output_dir, os.path.split(filenames[-1])[-1])))
        #         flag = True
        #         break
        # if os.path.exists(os.path.join(output_dir, os.path.split(filenames[-1])[-1])) and not flag:
        #     continue
        # if not flag:
        #     continue
        # run attack
        # adv = attacker.attack(model, img.cuda(), label_true.cuda(), label_target.cuda())
        adv_01 = attacker(img.cuda(), label_true.cuda())
        adv = attacker_02(adv_01.cuda(), label_true.cuda())
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