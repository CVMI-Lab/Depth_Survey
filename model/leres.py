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
        transform = transforms.Compose([transforms.ToTensor(),
		                                transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225) )])
        img = transform(img)
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
    return img

def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


class LeReS:
    def __init__(self, model_dir, ckpt_path, **kwargs):
        torch.cuda.init()  #
        torch.cuda.synchronize()  #
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        sys.path.append(model_dir)
        from lib.multi_depth_model_woauxi import RelDepthModel

        self.model = RelDepthModel(backbone='resnext101')
        self.load_ckpt(ckpt_path, self.model, None, None)
        self.model = self.model.to(self.device).eval()


    def load_ckpt(self, load_ckpt, depth_model, shift_model, focal_model):
        """
        Load checkpoint.
        """
        if os.path.isfile(load_ckpt):
            print("loading checkpoint %s" % load_ckpt)
            checkpoint = torch.load(load_ckpt)
            if shift_model is not None:
                shift_model.load_state_dict(strip_prefix_if_present(checkpoint['shift_model'], 'module.'),
                                        strict=True)
            if focal_model is not None:
                focal_model.load_state_dict(strip_prefix_if_present(checkpoint['focal_model'], 'module.'),
                                        strict=True)
            depth_model.load_state_dict(strip_prefix_if_present(checkpoint['depth_model'], "module."),
                                        strict=True)
            del checkpoint
            torch.cuda.empty_cache()

    def prepare_input(self, data):
        # change to PIL Image
        rgb_int = (data["image"].squeeze().numpy()).astype(np.uint8)  # [3, H, W]
        rgb_int = np.moveaxis(rgb_int, 0, -1)  # [H, W, 3]
        A_resize = cv2.resize(rgb_int, (448, 448))
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
        Forward pass through the DepthAnythingV2 model.
        """
        # Prepare input data
        img_torch, img_raw = self.prepare_input(data)

        with torch.no_grad():
            pred_depth = self.model.inference(img_torch).cpu().numpy().squeeze()
        pred_depth_ori = cv2.resize(pred_depth, (img_raw.shape[1], img_raw.shape[0]))

        return self.prepare_output(pred_depth_ori, data)