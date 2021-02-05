# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import os
import pretrainedmodels

def preprocess_image(img):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    img_tensor = preprocess(img)
    img_variable = Variable(img_tensor.unsqueeze(0))

    return img_variable

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    is_cuda = True
    # input image
    finalconv_name = 'layer4'
    # networks
    # net = models.resnet101(pretrained=True)
    net = pretrainedmodels.se_resnext101_32x4d(num_classes=1000, pretrained='imagenet')
    if is_cuda:
        net = net.cuda()
    net.eval()

    root_path = '/raid/chenby/tianchi/imagenet/images'
    save_root = '/raid/chenby/tianchi/imagenet/cam_mask_se_res101_4d_03'
    if not os.path.isdir(save_root):
        os.makedirs(save_root)
    image_names = sorted(os.listdir(root_path))[:]
    for index in range(len(image_names)):
        # hook the feature extractor
        features_blobs = []
        def hook_feature(module, input, output):
            features_blobs.append(output.data.cpu().numpy())
        net._modules.get(finalconv_name).register_forward_hook(hook_feature)

        # get the softmax weight
        params = list(net.parameters())
        weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

        image_path = os.path.join(root_path, image_names[index])
        # print(image_path)
        img_pil = Image.open(image_path)
        img_variable = preprocess_image(img=img_pil)
        if is_cuda:
            img_variable = img_variable.cuda()
        logit = net(img_variable)

        h_x = F.softmax(logit, dim=1).cpu().data.squeeze()
        probs, idx = h_x.sort(0, True)
        probs = probs.numpy()
        idx = idx.numpy()

        # output the prediction
        # for i in range(0, 5):
        #     print('{:.3f} -> {}'.format(probs[i], idx[i]))

        # generate class activation mapping for the top1 prediction
        CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
        # print(np.unique(CAMs[0]), np.max(CAMs[0]/255.0))
        CAMs[0] = np.uint8(CAMs[0]/255.0 > 0.3) * 255
        # print(np.unique(CAMs[0]), np.max(CAMs[0] / 255.0))
        # render the CAM and output
        print(f'output {image_names[index]} for the top1 prediction: {idx[0]}')
        img = cv2.imread(image_path)
        height, width, _ = img.shape
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
        # result = heatmap * 0.3 + img * 0.5
        result = cv2.resize(CAMs[0]//255,(width, height)) * 255
        # print(np.unique(result))

        save_path = os.path.join(save_root, image_names[index].replace('jpg', 'png'))
        cv2.imwrite(save_path, result)
        # cv2.imwrite('CAM.jpg', result)