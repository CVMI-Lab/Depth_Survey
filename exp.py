import os
import yaml

TASKS = [
    ["Metric3Dv2", "metric"],
    ["Metric3Dv2", "dep_lstq"],
    
    ["MiDasV31", "disp_lstq"],
    ["MiDasV31", "dep_lstq"],

    ["DiffE2EFT", "dep_lstq"],
    ["GenPercept", "dep_lstq"],
]

for task in TASKS:
    METHOD_NAME, METRIC_NAME = task

    # List of datasets to evaluate
    datasets = [
        "DDAD", "DIODE", "ETH3D", "GSO", "HAMMER",
        "iBims-1", "KITTI", "NYUv2", "Sintel", "Spring"
    ]

    # Template for the YAML config
    yaml_template = {
        "dataset": None,
        "dataset_params": {
            "path": None,
            "width": 640,
            "height": 480,
            "split": ".index.txt",
            "depth_unit": 1.0
        },
        "model_name": f"{METHOD_NAME}",
        "eval_depth": {
            "metric_names": [
                "Abs Rel",
                "delta < 1.25",
                "delta < 1.25^2",
                "delta < 1.25^3"
            ],
            "disp_input": False
        },
        "vis_depth": False,
        "save_dir": None
    }

    if METRIC_NAME == 'metric':
        yaml_template["eval_depth"]["metric_scale"] = True
        yaml_template["eval_depth"]["depth_alignment"] = "lstsq"
        yaml_template["eval_depth"]["disp_input"] = False
    elif METRIC_NAME == 'dep_lstq':
        yaml_template["eval_depth"]["metric_scale"] = False
        yaml_template["eval_depth"]["depth_alignment"] = "lstsq"
        yaml_template["eval_depth"]["disp_input"] = False
    elif METRIC_NAME == 'disp_lstq':
        yaml_template["eval_depth"]["metric_scale"] = False
        yaml_template["eval_depth"]["depth_alignment"] = "lstsq"
        yaml_template["eval_depth"]["disp_input"] = True

    if METHOD_NAME == "Metric3Dv2" or METHOD_NAME == "DiffE2EFT":
        # Specific parameters for Metric3Dv2
        yaml_template["model_params"] = {
            "pseudo_arg": "placeholder"  # Placeholder, adjust as needed
        }
    elif METHOD_NAME == "MiDasV31":
        # Specific parameters for MiDasV31
        yaml_template["model_params"] = {
            "model_dir": "/mnt/sda/hyh/depth/MiDaS",
            "model_ckpt_path": "/mnt/sda/hyh/depth/MiDaS/ckpt/dpt_beit_large_512.pt",
        }
    elif METHOD_NAME == "GenPercept":
        # Specific parameters for GenPercept
        yaml_template["model_params"] = {
            "model_dir": "/mnt/sda/hyh/depth/GenPercept",
            "pretrained_path": "/mnt/sda/hyh/depth/GenPercept/pretrained_weights/stable-diffusion-2-1",
            "unet_path": "/mnt/sda/hyh/depth/GenPercept/weights/genpercept-models/unet_depth_v2",
        }

    # Output directories
    config_dir = f"configs/moge_benchmark/{METHOD_NAME.lower()}"
    os.makedirs(config_dir, exist_ok=True)

    sh_lines = []

    for dataset in datasets:
        config = yaml_template.copy()
        config["dataset"] = dataset
        config["dataset_params"] = config["dataset_params"].copy()
        config["dataset_params"]["path"] = f"/mnt/sda/hyh/data//eval/{dataset}"
        config["save_dir"] = f"results/{METHOD_NAME.lower()}/eval_{METHOD_NAME.lower()}-{METRIC_NAME}_{dataset}"
        config_filename = f"{METHOD_NAME.lower()}-{METRIC_NAME}_{dataset}.yaml"
        config_path = os.path.join(config_dir, config_filename)
        # Write YAML file
        with open(config_path, "w") as f:
            yaml.dump(config, f, sort_keys=False)
        # Add command to shell script
        sh_lines.append(f"python eval.py configs/moge_benchmark/{METHOD_NAME.lower()}/{config_filename}")

    # Write shell script
    with open(f"run_all_evals_{METHOD_NAME.lower()}_metric_{METRIC_NAME}.sh", "w") as f:
        f.write("#!/bin/bash\n")
        for line in sh_lines:
            f.write(line + "\n")