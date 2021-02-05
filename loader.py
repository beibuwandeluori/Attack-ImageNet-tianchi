import os
import random
import torch
import numpy as np
import glob
import pandas as pd
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F


def get_transforms(size=500, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
	return transforms.Compose([
		transforms.Resize((size, size)),
		transforms.ToTensor(),
		# transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
		transforms.Normalize(mean=mean, std=std)
	])


def input_diversity(image, div_prob=0.5, low=200, high=500):
	if random.random() > div_prob:
		return image
	rnd = random.randint(low, high)
	rescaled = F.interpolate(image, size=[rnd, rnd], mode='bilinear')
	h_rem = high - rnd
	w_rem = high - rnd
	pad_top = random.randint(0, h_rem)
	pad_bottom = h_rem - pad_top
	pad_left = random.randint(0, w_rem)
	pad_right = w_rem - pad_left
	padded = F.pad(rescaled, [pad_top, pad_bottom, pad_left, pad_right], 'constant', 0)
	return padded


class ImageNet_A(Dataset):
	def __init__(self, root_dir, csv_name='dev.csv', folder_name='images',
				 use_target=True,
				 transforms=None,
				 mask_root_dir=None,
				 attack_background=False):
		# labels_dir = os.path.join(root_dir, csv_name)
		labels_dir = '/raid/chenby/tianchi/imagenet/' + csv_name
		self.image_dir = os.path.join(root_dir, folder_name)
		self.labels = pd.read_csv(labels_dir)
		self.use_target = use_target
		self.mask_root_dir = mask_root_dir
		self.attack_background = attack_background

		self.transforms = transforms

	def __len__(self):
		l = len(self.labels)
		return l

	def __getitem__(self, idx):
		filename = os.path.join(self.image_dir, self.labels.at[idx, 'ImageId'])
		in_img_t = cv2.imread(filename)[:, :, ::-1]
		if self.transforms is not None:
			img = self.transforms(Image.fromarray(in_img_t))
		else:
			# in_img_t = cv2.resize(in_img_t, (224, 224))
			in_img = np.transpose(in_img_t.astype(np.float32), axes=[2, 0, 1])
			img = in_img / 255.0

		label_true = self.labels.at[idx, 'TrueLabel']
		if self.mask_root_dir is not None:
			mask_path = filename.replace(self.image_dir, self.mask_root_dir).replace('jpg', 'png')
			mask = cv2.imread(mask_path)[:, :, ::-1]
			mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)//255
			if self.attack_background:
				mask = 1 - mask
			mask = np.expand_dims(mask, axis=0)  # 扩展一个维度

			if self.use_target:
				label_target = 999 - label_true
				return img, mask, label_true, label_target, filename
			else:
				return img, mask, label_true, filename
		if self.use_target:
			# label_target = self.labels.at[idx, 'TargetClass']
			label_target = 999 - label_true

			return img, label_true, label_target, filename
		else:
			return img, label_true, filename


if __name__ == '__main__':
	low = 200
	high = 500
	use_target = False
	root_dir = '/raid/chenby/tianchi/imagenet'
	# root_dir = '/data1/cby/py_project/Attack-ImageNet/results/02_ensemble_nt/'
	mask_root_dir = '/raid/chenby/tianchi/imagenet/cam_mask_res101'
	dataset = ImageNet_A(root_dir, csv_name='dev.csv', folder_name='images',  use_target=use_target,
						 transforms=None, mask_root_dir=mask_root_dir, attack_background=True)
	# print(len(dataset))
	train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
	if not use_target:
		for i, (img, mask, label, _) in enumerate(train_loader):
			img = input_diversity(img, low=low, high=high)
			print(i, '/', len(train_loader), img.shape, mask.shape, label.shape, label[0])
			print(torch.max(img), torch.unique(mask), (mask * img).shape)
			if i == 20:
				break
	else:
		for i, (img, label, label_target, _) in enumerate(train_loader):
			print(i, '/', len(train_loader), img.shape, label.shape, label_target.shape, label[0], label_target[0])
			if i == 20:
				break
