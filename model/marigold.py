import sys
import os
import numpy as np
import torch
from PIL import Image

class Marigold:
    def __init__(self, model_dir, ckpt_path, half_precision, **kwargs):
        torch.cuda.init()  # 
        torch.cuda.synchronize()  # 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        sys.path.append(model_dir)
        from marigold import MarigoldDepthPipeline

        # half resolution
        if half_precision:
            dtype = torch.float16
            variant = "fp16"
            print(
                f"Running with half precision ({dtype}), might lead to suboptimal result."
        )
        else:
            dtype = torch.float32
            variant = None

        self.pipe = MarigoldDepthPipeline.from_pretrained(
        ckpt_path, variant=variant, torch_dtype=dtype
    )
        
        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.to(self.device)

        # --- denoise parameters ---
        self.denoise_steps = kwargs.get("denoise_steps", 1)
        self.ensemble_size = kwargs.get("ensemble_size", 1)
        self.processing_res = kwargs.get("processing_res", 0)
        self.match_input_res = not kwargs.get("output_processing_res", False)
        self.resample_method = kwargs.get("resample_method", "bilinear")
        self.seed = kwargs.get("seed", None)
        self.color_map = kwargs.get("color_map", None)

        # Print out config
        print(
            f"Inference settings: checkpoint = `{ckpt_path}`, "
            f"with denoise_steps = {self.denoise_steps or self.pipe.default_denoising_steps}, "
            f"ensemble_size = {self.ensemble_size}, "
            # f"processing resolution = {self.processing_res or self.pipe.default_processing_resolution}, "
            f"seed = {self.seed}; "
            f"color_map = {self.color_map}."
        )


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
        input_image = self.prepare_input(data)

        # Random number generator
        if self.seed is None:
            generator = None
        else:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.seed)

        with torch.no_grad():
            # Perform inference
            pipe_out = self.pipe(
                input_image,
                denoising_steps=self.denoise_steps,
                ensemble_size=self.ensemble_size,
                processing_res=self.processing_res,
                match_input_res=self.match_input_res,
                batch_size=1,
                color_map=self.color_map,
                show_progress_bar=True,
                resample_method=self.resample_method,
                generator=generator,
            )

        output = self.prepare_output(pipe_out, data)

        return output
