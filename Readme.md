# Monocular Depth Estimation Survey

[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](xxx)
[![Website](assets/badge-website.svg)](xxx)

In this repo, we provide a unified framework for **Monocular Depth Estimation**.

Our framework provides a convenient interface for **various dataset** and **various methods**, which supports a fair comparison by aligning the output and evaluation scripts.


### Evaluation 
The dataset, model and evaluation metric configuration can be set in the yaml file in [configs](./configs/). E.g.,

- Dataset Config:
  ```
  dataset: NYUv2
  dataset_params:
    path: /mnt/pfs/data/RGBD/moge_eval/NYUv2
    width: 640
    height: 480
    split: ".index.txt"
    depth_unit: 1.0
  ```
- Model Config:
  ```
  model_name: "Marigold"
  model_params:
    model_dir: "/mnt/pfs/users/sunyangtian/Depth/Marigold"
    ckpt_path: "/mnt/pfs/share/pretrained_model/marigold-depth-v1-1"
    denoise_steps: 1
    ensemble_size: 1
    half_precision: False
    processing_res: 0
    output_processing_res: False
    resample_method: bilinear
    color_map: Spectral
  ```
- Metric Config
  ```
  eval_depth:
    metric_names: 
      - 'Abs Rel'
      - 'delta < 1.25'
      - 'delta < 1.25^2'
      - 'delta < 1.25^3'
    depth_alignment: "lstsq"
    metric_scale: False
  ```
- Output Config
  ```
  vis_depth: True
  save_dir: debug_marigold_nyuv2
  ```

Finally, the evaluation process can be performed by
  ```
    python eval.py configs/moge_benchmark/marigold/marigold_nyuv2.yaml
  ```

You can also use `eval_all.sh` to evaluate all datasets with one command.

### Supported Datasets
Please refer to [dataset](./dataset/Readme.md) for more details.


### Supported Methods
Please refer to [model](./model/Readme.md) for more details.

### Acknowledgements
