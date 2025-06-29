import sys
import os
import numpy as np
import torch
import cv2

class DPT:
    def __init__(self, model_dir, ckpt_path, optimize=True, **kwargs):
        torch.cuda.init()  #
        torch.cuda.synchronize()  #
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        sys.path.append(model_dir)
        from dpt.models import DPTDepthModel
        from dpt.midas_net import MidasNet_large
        from dpt.transforms import Resize, NormalizeImage, PrepareForNet
        from torchvision.transforms import Compose

        net_w = net_h = 384
        self.model = DPTDepthModel(
            path=ckpt_path,
            backbone="vitl16_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        self.optimize = optimize

        self.transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

        self.model.eval()

        if optimize == True and self.device == torch.device("cuda"):
            self.model = self.model.to(memory_format=torch.channels_last)
            self.model = self.model.half()

        self.model.to(self.device)


    def prepare_input(self, data):
        # change to PIL Image
        rgb_int = (data["image"].squeeze().numpy()).astype(np.uint8)  # [3, H, W]
        rgb_int = np.moveaxis(rgb_int, 0, -1)  # [H, W, 3]
        img_input = self.transform({"image": rgb_int / 255.0})["image"]
        return img_input, rgb_int

    def prepare_output(self, pred, data):
        # disp_min = pred.min()
        # disp_max = pred.max()
        # disp = (pred - disp_min) / (disp_max - disp_min)

        # depth = 1 - disp
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
        Forward pass through the DepthAnythingV2 model.
        """
        # Prepare input data
        img, img_raw = self.prepare_input(data)

        with torch.no_grad():
            sample = torch.from_numpy(img).to(self.device).unsqueeze(0)

            if self.optimize == True and self.device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()

            prediction = self.model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_raw.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )  # [H,W], numpy

        return self.prepare_output(prediction, data)