import os
from configs.config_utils import import_class_from_module, parse_metric_config
from metrics import MetricsManager, depth_evaluation_wrap
from omegaconf import OmegaConf
from utils.vis_utils import save_rgb_depth
from dataset import EvalDataLoaderPipeline
import sys


if __name__ == "__main__":
    config_path = sys.argv[1]
    config = OmegaConf.load(config_path)
    benchmark_name, benchmark_config = config['dataset'], config['dataset_params']

    ### model
    model_cls = import_class_from_module("model", config["model_name"])
    model = model_cls(**config["model_params"])

    ### metrics
    metric_names = parse_metric_config(config)
    metrics_manager = MetricsManager(metric_names=metric_names)

    save_dir = config.get("save_dir", "./debug_output_test")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"metrics.csv")

    
    with (
        EvalDataLoaderPipeline(**benchmark_config) as eval_data_pipe,
    ):  
        for data_idx in range(len(eval_data_pipe)):
            data = eval_data_pipe.get()

            seq = f"{data_idx:03d}_{data['seq_name']}"
            print("processing seq:", seq)

            # Forward pass through the model
            output = model.forward(data)

            ### metric
            metric = {"seq_name" : seq}

            ### depth
            if 'eval_depth' in config:
                pred_depth = output["pred_disp"] if config["eval_depth"].get("disp_input", False) else output["pred_depth"]
                res = depth_evaluation_wrap(predicted_depth_original=pred_depth, ground_truth_depth_original=data['depth'], custom_mask=data['depth_mask'], **config["eval_depth"])
                print(res[0])
                metric.update(res[0])

            if config.get('vis_depth', False):
                ### visualization
                save_depth_path = os.path.join(save_dir, seq.replace("/", "_"))
                save_depth_path = save_depth_path[:-4] + ".webp"
                rgb_img = data['image'].numpy().transpose(1, 2, 0)  # [H,W,3]

                save_rgb_depth(
                    rgb=data['image'].permute(1,2,0).numpy()/255., 
                    depth=output["pred_depth"],
                    save_path=save_depth_path,
                    )
                
            ######## update metric
            metrics_manager.update_metrics(metric)
            metrics_manager.export_to_csv(save_path)