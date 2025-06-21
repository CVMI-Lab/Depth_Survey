import sys
import os
import numpy as np
import torch

class DepthAnythingV2:
    def __init__(self, model_dir, ckpt_path, encoder='vitl', **kwargs):
        torch.cuda.init()  # 
        torch.cuda.synchronize()  # 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        sys.path.append(model_dir)
        from depth_anything_v2.dpt import DepthAnythingV2

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        self.model = DepthAnythingV2(**model_configs[encoder])
        self.model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        self.model = self.model.to(self.device).eval()


    def prepare_input(self, data):
        # change to PIL Image
        rgb_int = (data["image"].squeeze().numpy()).astype(np.uint8)  # [3, H, W]
        rgb_int = np.moveaxis(rgb_int, 0, -1)  # [H, W, 3]
        return rgb_int
    
    def prepare_output(self, depth, data):
        output = {
            'pred_depth': depth, # [H, W]
        }
        return output

    def forward(self, data):
        """
        Forward pass through the DepthAnythingV2 model.
        """
        # Prepare input data
        raw_img = self.prepare_input(data)

        depth = self.model.infer_image(raw_img)  # [H,W], numpy

        return self.prepare_output(depth, data)