import sys
import os
import numpy as np
import torch
from PIL import Image

class GeoWizard:
    def __init__(self, model_dir, ckpt_path, **kwargs):
        torch.cuda.init()  #
        torch.cuda.synchronize()  #
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        sys.path.append(os.path.join(model_dir, "geowizard"))
        from models.geowizard_pipeline import DepthNormalEstimationPipeline
        from diffusers import DiffusionPipeline, DDIMScheduler, AutoencoderKL
        from models.unet_2d_condition import UNet2DConditionModel
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        from utils.seed_all import seed_all

        vae = AutoencoderKL.from_pretrained(ckpt_path, subfolder='vae')
        scheduler = DDIMScheduler.from_pretrained(ckpt_path, subfolder='scheduler')
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(ckpt_path, subfolder="image_encoder")
        feature_extractor = CLIPImageProcessor.from_pretrained(ckpt_path, subfolder="feature_extractor")
        unet = UNet2DConditionModel.from_pretrained(ckpt_path, subfolder="unet")

        self.pipe = DepthNormalEstimationPipeline(vae=vae,
                                image_encoder=image_encoder,
                                feature_extractor=feature_extractor,
                                unet=unet,
                                scheduler=scheduler)

        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except:
            pass  # run without xformers

        self.pipe = self.pipe.to(self.device)

        # --- denoise parameters ---
        self.denoise_steps = kwargs.get("denoise_steps", 10)
        self.ensemble_size = kwargs.get("ensemble_size", 10)
        self.processing_res = kwargs.get("processing_res", 768)
        self.match_input_res = not kwargs.get("output_processing_res", False)
        self.domain = kwargs.get("domain", "indoor")
        self.color_map = kwargs.get("color_map", "Spectral")

        seed_all(0)


    def prepare_input(self, data):
        # change to PIL Image
        rgb_int = (data["image"].squeeze().numpy()).astype(np.uint8)  # [3, H, W]
        rgb_int = np.moveaxis(rgb_int, 0, -1)  # [H, W, 3]
        input_image = Image.fromarray(rgb_int)
        return input_image

    def prepare_output(self, pipe_out, data):
        depth_pred: np.ndarray = pipe_out.depth_np
        depth_colored: Image.Image = pipe_out.depth_colored

        output = {
            'pred_depth' : depth_pred,  # [H, W]
            'pred_depth_colored' : depth_colored,
        }

        return output

    def forward(self, data):
        """
        Forward pass through the DepthAnythingV2 model.
        """
        # Prepare input data
        input_image = self.prepare_input(data)

        # import ipdb; ipdb.set_trace()

        # predict the depth & normal here
        with torch.no_grad():
            pipe_out = self.pipe(input_image,
                denoising_steps = self.denoise_steps,
                ensemble_size= self.ensemble_size,
                processing_res = self.processing_res,
                match_input_res = self.match_input_res,
                domain = self.domain,
                color_map = self.color_map,
                show_progress_bar = True,
            )

        return self.prepare_output(pipe_out, data)