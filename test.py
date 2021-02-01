# Helper function for extracting features from pre-trained models
import sys, os
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from loader import ImageNet_A, get_transforms
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
        logits = (logits1 + logits2) / 2

        return logits


def load_model():
    pretrained_model = resnet101_denoise()
    loaded_state_dict = torch.load(os.path.join('weight', 'Adv_Denoise_Resnext101.pytorch'))
    pretrained_model.load_state_dict(loaded_state_dict, strict=True)

    # pretrained_model = resnet152_denoise()
    # loaded_state_dict = torch.load(os.path.join('weight', 'Adv_Denoise_Resnet152.pytorch'))
    # pretrained_model.load_state_dict(loaded_state_dict)

    # pretrained_model = resnet152()
    # loaded_state_dict = torch.load(os.path.join('weight', 'Adv_Resnet152.pytorch'))
    # pretrained_model.load_state_dict(loaded_state_dict)
    model = nn.Sequential(
        # Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        Permute([2, 1, 0]),
        pretrained_model
    )

    return model


def eval_model(model, eval_loader):
    model.eval()
    n_correct = 0
    labels = np.array([])
    preds = np.array([])
    outputs = np.array([])
    with torch.no_grad():
        for i, (img, label, _) in enumerate(eval_loader):
            img, label = Variable(img.cuda()), Variable(label.cuda())
            # print(img.size(), label.size())
            class_output = model(img)
            class_output = nn.Softmax(dim=1)(class_output)

            pred = torch.max(class_output, 1)
            n_correct += (pred[1] == label).sum().item()

            labels = np.concatenate([labels, label.data.cpu().numpy()], axis=0)
            preds = np.concatenate([preds, pred[1].data.cpu().numpy()], axis=0)
            outputs = np.concatenate([outputs, 1-class_output.data.cpu().numpy()[:, 0]], axis=0)

            if i % 10 == 0:
                print(i+1, 'current_correct:', n_correct, 'current_len:', (i+1) * batch_size, len(eval_loader.dataset))

    accu = float(n_correct) / len(eval_loader.dataset) * 100
    print('Classes accuracy : {:.4f}-{:.4f}'.format(accu, 100-accu))


if __name__ == '__main__':
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # mean = [0.5, 0.5, 0.5]
    # std = [0.5, 0.5, 0.5]

    input_dir = '/raid/chenby/tianchi/imagenet/'
    # input_dir = '/data1/cby/py_project/Attack-ImageNet/results/04_ensemble_MIM_div_step100_8_iter2_step50_4/'
    batch_size = 32
    size = 456  # 456
    size_2 = 224

    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    # ensemble model
    # model = load_model()
    model = model_selection(model_name='efficientnet-b5', advprop=False)  # efficientnet-b5
    model = nn.Sequential(
        Resize(input_size=[size, size]),
        Normalize(mean, std),
        model
    )
    # model2 = model_selection(model_name='resnet50', advprop=False)
    # model2 = nn.Sequential(
    #     Resize(input_size=[size_2, size_2]),
    #     Normalize(mean, std),
    #     model2
    # )
    # model = Ensemble(model, model2)

    model.cuda()
    model.eval()

    # set dataset
    # dataset = ImageNet_A(input_dir, use_target=False, transforms=get_transforms(size, mean, std))
    dataset = ImageNet_A(input_dir, use_target=False, transforms=None)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=False)
    eval_model(model, loader)
