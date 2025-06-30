import sys
import os.path as osp
import numpy as np
import torch
import cv2
from collections import OrderedDict
import torchvision.transforms as transforms

try:
    import diffusers
    from diffusers import DiffusionPipeline
    from diffusers import UNet2DConditionModel
except:
    print("No diffusers package found, please install it with `pip install diffusers`")
    pass

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


class GenPercept:
    def __init__(self, model_dir, pretrained_path, unet_path, **kwargs):
        torch.cuda.init()  #
        torch.cuda.synchronize()  #
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        sys.path.append(model_dir)
        from safetensors.torch import load_model, save_model, load_file
        from src.customized_modules.ddim import DDIMSchedulerCustomized
        from genpercept import GenPerceptPipeline

        pre_loaded_dict = dict()
        dtype = torch.float32
        genpercept_pipeline = True
        variant = None
        
        unet = UNet2DConditionModel.from_pretrained(pretrained_path, subfolder='unet')
        if osp.exists(osp.join(unet_path, 'diffusion_pytorch_model.bin')):
            unet_ckpt_path = osp.join(unet_path, 'diffusion_pytorch_model.bin')
        elif osp.exists(osp.join(unet_path, 'diffusion_pytorch_model.safetensors')):
            unet_ckpt_path = osp.join(unet_path, 'diffusion_pytorch_model.safetensors')
        ckpt = load_file(unet_ckpt_path)
        unet.load_state_dict(ckpt)
        pre_loaded_dict['unet'] = unet.to(dtype=dtype).to(device=self.device)

        pre_loaded_dict['scheduler'] = DDIMSchedulerCustomized.from_pretrained(f'{model_dir}/hf_configs/scheduler_beta_1.0_1.0')
        pipe: GenPerceptPipeline = GenPerceptPipeline.from_pretrained(
            "prs-eth/marigold-v1-0", variant=variant, torch_dtype=dtype, genpercept_pipeline=genpercept_pipeline, **pre_loaded_dict
        )
        pipe = pipe.to(self.device)
        self.model = pipe
        
        self.model = self.model.to(self.device)

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
            pred_depth = self.model(
                img_torch.to(self.device) * 255,
                denoising_steps=1,
                ensemble_size=1,
                processing_res=None,
                match_input_res=True,
                batch_size=1,
                color_map='Spectral',
                show_progress_bar=True,
                resample_method="bilinear",
                generator=None,
                mode="depth",
            )
            pred_depth = pred_depth.pred_np.squeeze()
        pred_depth_ori = cv2.resize(pred_depth, (img_raw.shape[1], img_raw.shape[0]))
        return self.prepare_output(pred_depth_ori, data)
