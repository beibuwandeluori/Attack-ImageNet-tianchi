from typing import Optional, Tuple
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchattacks


def input_diversity(image, div_prob=0.9, low=200, high=500):
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


class Attacker:
    def __init__(self,
                 steps: int,
                 quantize: bool = True,
                 levels: int = 256,
                 max_norm: Optional[float] = None,
                 div_prob: float = 0.9,
                 loss_amp: float = 4.0,
                 device: torch.device = torch.device('cpu'),
                 low: int = 270,
                 high: int = 299) -> None:
        self.steps = steps

        self.quantize = quantize
        self.levels = levels
        self.max_norm = max_norm
        self.div_prob = div_prob
        self.loss_amp = loss_amp
        self.low = low
        self.high = high
        
        self.device = device

    def input_diversity(self, image):
        if random.random() > self.div_prob:
            return image
        rnd = random.randint(self.low, self.high)
        rescaled = F.interpolate(image, size=[rnd, rnd], mode='bilinear')
        h_rem = self.high - rnd
        w_rem = self.high - rnd
        pad_top = random.randint(0, h_rem)
        pad_bottom = h_rem - pad_top
        pad_left = random.randint(0, w_rem)
        pad_right = w_rem - pad_left
        padded = F.pad(rescaled, [pad_top, pad_bottom, pad_left, pad_right], 'constant', 0)
        return padded

    def attack(self, 
               model: nn.Module, 
               inputs: torch.Tensor, 
               labels_true: torch.Tensor,
               labels_target: torch.Tensor)-> torch.Tensor:

        batch_size = inputs.shape[0]
        delta = torch.zeros_like(inputs, requires_grad=True)

        # setup optimizer
        optimizer = optim.SGD([delta], lr=1, momentum=0.9)

        # for choosing best results
        best_loss = 1e4 * torch.ones(inputs.size(0), dtype=torch.float, device=self.device)
        best_delta = torch.zeros_like(inputs)

        for _ in range(self.steps):
            if self.max_norm:
                delta.data.clamp_(-self.max_norm, self.max_norm)
                if self.quantize:
                    delta.data.mul_(self.levels - 1).round_().div_(self.levels - 1)

            adv = inputs + delta
            div_adv = self.input_diversity(adv)

            logits = model(div_adv)

            ce_loss_true = F.cross_entropy(logits, labels_true, reduction='none')     
            # ce_loss_target = F.cross_entropy(logits, labels_target, reduction='none')

            # fuse targeted and untargeted
            # loss = self.loss_amp * ce_loss_target - ce_loss_true
            loss = - ce_loss_true
            
            is_better = loss < best_loss

            best_loss[is_better] = loss[is_better]
            best_delta[is_better] = delta.data[is_better]
            
            loss = torch.mean(loss)
            optimizer.zero_grad()
            loss.backward()

            # renorm gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=float('inf'), dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))

            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])

            optimizer.step()

            # avoid out of bound
            delta.data.add_(inputs)
            delta.data.clamp_(0, 1).sub_(inputs)

        return inputs + best_delta


class AttackerPGD(torchattacks.PGD):
    def __init__(self, model, eps=0.3, alpha=2 / 255, steps=40, random_start=False, low=200, high=500, div_prob=0.9):
        super(AttackerPGD, self).__init__(model, eps=eps, alpha=alpha, steps=steps, random_start=random_start)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.low = low
        self.high = high
        self.div_prob = div_prob

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        labels = self._transform_label(images, labels)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for i in range(self.steps):
            adv_images.requires_grad = True
            # adv_div = self.input_diversity(adv_images)  # 数据增强
            adv_div = input_diversity(adv_images, div_prob=self.div_prob, low=self.low, high=self.high)
            outputs = self.model(adv_div)

            cost = self._targeted * loss(outputs, labels)

            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]
            # grad = grad * mask
            adv_images = adv_images.detach() - self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

    def input_diversity(self, image):
        if random.random() > self.div_prob:
            return image
        rnd = random.randint(self.low, self.high)
        rescaled = F.interpolate(image, size=[rnd, rnd], mode='bilinear')
        h_rem = self.high - rnd
        w_rem = self.high - rnd
        pad_top = random.randint(0, h_rem)
        pad_bottom = h_rem - pad_top
        pad_left = random.randint(0, w_rem)
        pad_right = w_rem - pad_left
        padded = F.pad(rescaled, [pad_top, pad_bottom, pad_left, pad_right], 'constant', 0)
        return padded


class AttackerTPGD(torchattacks.TPGD):
    def __init__(self, model, eps=8/255.0, alpha=2/255.0, steps=7, low=200, high=500, div_prob=0.9):
        super(AttackerTPGD, self).__init__(model, eps, alpha, steps)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self._attack_mode = 'only_default'
        self.low = low
        self.high = high
        self.div_prob = div_prob

    def forward(self, images, labels=None):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        logit_ori = self.model(images).detach()

        adv_images = images + 0.001 * torch.randn_like(images)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        loss = nn.KLDivLoss(reduction='sum')

        for i in range(self.steps):
            adv_images.requires_grad = True
            # adv_div = self.input_diversity(adv_images)  # 数据增强
            adv_div = input_diversity(adv_images, div_prob=self.div_prob, low=self.low, high=self.high)
            logit_adv = self.model(adv_div)

            cost = loss(F.log_softmax(logit_adv, dim=1),
                        F.softmax(logit_ori, dim=1))

            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


class AttackerMIFGSM(torchattacks.MIFGSM):
    def __init__(self, model, eps=8 / 255, steps=5, decay=1.0, low=200, high=500, div_prob=0.9):
        super(AttackerMIFGSM, self).__init__(model, eps, steps, decay)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = self.eps / self.steps
        self.low = low
        self.high = high
        self.div_prob = div_prob

    def forward(self, images, labels, mask=None):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        labels = self._transform_label(images, labels)

        loss = nn.CrossEntropyLoss()
        momentum = torch.zeros_like(images).detach().to(self.device)

        adv_images = images.clone().detach()

        for i in range(self.steps):
            adv_images.requires_grad = True
            # outputs = self.model(adv_images)
            # adv_div = self.input_diversity(adv_images)  # 数据增强
            adv_div = input_diversity(adv_images, div_prob=self.div_prob, low=self.low, high=self.high)
            outputs = self.model(adv_div)

            cost = self._targeted * loss(outputs, labels)

            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            grad_norm = torch.norm(nn.Flatten()(grad), p=1, dim=1)
            grad = grad / grad_norm.view([-1] + [1] * (len(grad.shape) - 1))
            grad = grad + momentum * self.decay
            momentum = grad
            if mask is not None:
                grad = grad * mask
            adv_images = adv_images.detach() - self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

