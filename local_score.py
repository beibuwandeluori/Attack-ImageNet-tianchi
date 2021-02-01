import torch
import numpy as np
from torch_fidelity import calculate_metrics
import lpips
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_image(path):
    if path[-3:] == 'dng':
        import rawpy
        with rawpy.imread(path) as raw:
            img = raw.postprocess()
    elif path[-3:] == 'bmp' or path[-3:] == 'jpg' or path[-3:] == 'png':
        import cv2
        return cv2.imread(path)[:, :, ::-1]
    else:
        img = (255*plt.imread(path)[:, :, :3]).astype('uint8')

    return img


def im2tensor(image, imtype=np.uint8, cent=1., factor=255./2.):
# def im2tensor(image, imtype=np.uint8, cent=1., factor=1.):
    return torch.Tensor((image / factor - cent)[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))


def get_fid(src1_path, src2_path):
    metrics_dict = calculate_metrics(src1_path, src2_path, cuda=True, fid=True)
    return metrics_dict['frechet_inception_distance']


def get_perception_distance(src1_path, src2_path):
    files = os.listdir(src1_path)
    # f = open("perception_dist.txt", 'w')
    files = tqdm(files)
    loss_fn_vgg = lpips.LPIPS(net='vgg', version=0.1)
    loss_fn_vgg.cuda()
    vgg_dist = 0.0
    for i, file in enumerate(files):
        img0 = im2tensor(load_image(os.path.join(src1_path, file)))
        img1 = im2tensor(load_image(os.path.join(src2_path, file)))
        img0 = img0.cuda()
        img1 = img1.cuda()
        d = loss_fn_vgg(img0, img1)
        vgg_dist += d.view(-1).item()
        # print('%s: %.3f' % (file, d))
        files.set_description("%s: %.3f" % (file, d))

        # f.writelines('%s: %.6f\n' % (file, d))
    # f.close()
    return vgg_dist/len(files)


def get_local_score(attack_acc, src1_path, src2_path):
    """

    :param attack_acc: 对抗样本对模型攻击准确率
    :param src1_path: 原图目录
    :param src2_path: 对抗样本目录
    :return: 每一项指标和最终本地得分结果
    """
    fid_dis = get_fid(src1_path, src2_path)
    perception_dis = get_perception_distance(src1_path, src2_path)
    score_fid = np.sqrt(1 - min(fid_dis, 200.)/200.)
    score_lpips = np.sqrt(1 - 2*(min(max(perception_dis, 0.2), 0.7) - 0.2))
    final_socre = 100 * attack_acc * score_fid * score_lpips
    print("attack_acc %.4f", attack_acc)
    print("fid_distance: %.4f, score_fid: %.4f" % (fid_dis, score_fid))
    print("perception_distance: %.4f, score_lpips: %.4f" % (perception_dis, score_lpips))
    print("final_score = attack_acc * score_fid * score_lpips = %.4f" % final_socre)
    return final_socre


if __name__ == '__main__':
    src1_path = "/raid/chenby/tianchi/imagenet/images/"  # 原图
    src2_path = "/data1/cby/py_project/Attack-ImageNet/results/04_ensemble_MIM_div_step100_8_iter2_step50_4/images/"  # 对抗样本
    attack_acc = 0.9  # 对抗样本对模型攻击准确率
    get_local_score(attack_acc, src1_path, src2_path)
