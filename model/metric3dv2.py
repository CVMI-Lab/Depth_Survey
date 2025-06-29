import sys
import os
import numpy as np
import torch
import cv2
from collections import OrderedDict
import torchvision.transforms as transforms


def scale_torch(img):
    """
    Scale the image and output it in torch.tensor.
    :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]
    if img.shape[2] == 3:
        transform = transforms.Compose([transforms.ToTensor()])
        img = transform(img)
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
    return img


class Metric3Dv2:
    def __init__(self, **kwargs):
        torch.cuda.init()  #
        torch.cuda.synchronize()  #
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_giant2', pretrain=True)
        self.model = self.model.to(self.device).eval()

    def prepare_input(self, data):
        # change to PIL Image
        rgb_int = (data["image"].squeeze().numpy()).astype(np.uint8)  # [3, H, W]
        rgb_int = np.moveaxis(rgb_int, 0, -1)  # [H, W, 3]
        A_resize = rgb_int
        img_torch = scale_torch(A_resize)[None, :, :, :]
        return img_torch, rgb_int

    def prepare_output(self, pred, data):
        depth = pred/pred.max()

        output = {
            # 'pred_disp': disp,  # [H, W],
            'pred_depth': depth # [H, W]
        }
        return output

    def forward(self, data):
        """
        Forward pass through the Metric3DV2 model.
        """
        # Prepare input data
        img_torch, img_raw = self.prepare_input(data)

        with torch.no_grad():
            pred_depth, _, _ = self.model.inference({'input': img_torch.cuda()})
            pred_depth = pred_depth.cpu().numpy().squeeze()
        pred_depth_ori = cv2.resize(pred_depth, (img_raw.shape[1], img_raw.shape[0]))

        return self.prepare_output(pred_depth_ori, data)

