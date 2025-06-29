import sys
import os
import numpy as np
import torch
import cv2
from collections import OrderedDict
import torchvision.transforms as transforms


first_execution = True
def process(device, model, model_type, image, input_size, target_size, optimize, use_camera):
    """
    Run the inference and interpolate.

    Args:
        device (torch.device): the torch device used
        model: the model used for inference
        model_type: the type of the model
        image: the image fed into the neural network
        input_size: the size (width, height) of the neural network input (for OpenVINO)
        target_size: the size (width, height) the neural network output is interpolated to
        optimize: optimize the model to half-floats on CUDA?
        use_camera: is the camera used?

    Returns:
        the prediction
    """
    global first_execution

    if "openvino" in model_type:
        if first_execution or not use_camera:
            print(f"    Input resized to {input_size[0]}x{input_size[1]} before entering the encoder")
            first_execution = False

        sample = [np.reshape(image, (1, 3, *input_size))]
        prediction = model(sample)[model.output(0)][0]
        prediction = cv2.resize(prediction, dsize=target_size,
                                interpolation=cv2.INTER_CUBIC)
    else:
        if type(image) is np.ndarray:
            sample = torch.from_numpy(image).to(device).unsqueeze(0)
        else:
            sample = image.to(device)

        if optimize and device == torch.device("cuda"):
            if first_execution:
                print("  Optimization to half-floats activated. Use with caution, because models like Swin require\n"
                    "  float precision to work properly and may yield non-finite depth values to some extent for\n"
                    "  half-floats.")
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        if first_execution or not use_camera:
            height, width = sample.shape[2:]
            print(f"    Input resized to {width}x{height} before entering the encoder")
            first_execution = False

        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=target_size[::-1],
                mode="bicubic",
                align_corners=False,
            )
        )

    return prediction


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


class MiDasV31:
    def __init__(self, model_dir, model_ckpt_path, **kwargs):
        torch.cuda.init()  #
        torch.cuda.synchronize()  #
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        sys.path.append(model_dir)
        from midas.model_loader import default_models, load_model

        model, transform, net_w, net_h = load_model(self.device, model_ckpt_path, "dpt_beit_large_512", optimize=False, height=None, square=False)
        
        self.model = model
        self.model = self.model.to(self.device).eval()
        self.transform = transform

        self.process_args = {
            "device": self.device,
            "model": self.model,
            "model_type": "dpt_beit_large_512",
            "input_size": (net_w, net_h),
            "target_size": (512, 512),  # Output size for interpolation
            "optimize": False,
            "use_camera": False
        }

    def prepare_input(self, data):
        # change to PIL Image
        rgb_int = (data["image"].squeeze().numpy()).astype(np.uint8)  # [3, H, W]
        rgb_int = np.moveaxis(rgb_int, 0, -1)  # [H, W, 3]
        A_resize = cv2.resize(rgb_int, (512, 512))
        img_torch = scale_torch(A_resize)[None, :, :, :]
        return img_torch, rgb_int
    
    def prepare_output(self, pred, data):
        disp = pred/pred.max()
        depth = 1 / (disp + 0.1)
        depth = (depth - depth.min()) / (depth.max() - depth.min())  # normalize to [0, 1]

        output = {
            'pred_disp': disp,  # [H, W],
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
            pred_depth = process(image=img_torch,  **self.process_args)
            pred_depth = pred_depth.cpu().numpy().squeeze()

        pred_depth_ori = cv2.resize(pred_depth, (img_raw.shape[1], img_raw.shape[0]))
        return self.prepare_output(pred_depth_ori, data)

