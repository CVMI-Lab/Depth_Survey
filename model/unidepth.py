import sys
import os
import numpy as np
import torch

class UniDepth:
    def __init__(self, model_dir, ckpt_path, **kwargs):
        torch.cuda.init()  # 
        torch.cuda.synchronize()  # 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        sys.path.append(model_dir)
        from unidepth.models import UniDepthV1

        self.model = UniDepthV1.from_pretrained(ckpt_path)
        self.model = self.model.to(self.device)


    def prepare_input(self, data):
        # change to PIL Image
        rgb = data["image"] / 255.
        return rgb  # [3,H,W]
    

    def prepare_output(self, predictions, data):
        depth = predictions["depth"]
        output = {
            'pred_depth': depth[0,0].cpu().numpy(),  # [1, H, W]
        }
        return output
    
    
    def forward(self, data):
        """
        Forward pass through the UniDepth model.
        """
        # Prepare input data
        rgb = self.prepare_input(data)

        predictions = self.model.infer(rgb)

        return self.prepare_output(predictions, data)



