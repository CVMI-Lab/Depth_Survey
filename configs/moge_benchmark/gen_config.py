import yaml
import json
import os

def load_datasets_info(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def generate_yaml_config(method_name, dataset_name, dataset_info, output_dir):
    config = {
        
        'dataset': dataset_name,
        'dataset_params': dataset_info,


        'model_name': 'DepthAnythingV2',
        'model_params': {
            'model_dir': '/mnt/pfs/users/sunyangtian/Depth/Depth-Anything-V2',
            'ckpt_path': '/mnt/pfs/share/pretrained_model/depth_anything_v2_vitl.pth',
            'encoder': 'vitl'
        },


        'eval_depth': {
            'metric_names': [
                'Abs Rel',
                'delta < 1.25',
                'delta < 1.25^2',
                'delta < 1.25^3'
            ],
            'depth_alignment': 'lstsq',
            'metric_scale': False,
            'disp_input': True
        },


        'vis_depth': False,
        'save_dir': f'eval_{method_name}_{dataset_name}'
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f'{method_name}_{dataset_name}.yaml')
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return output_path

def main():
    json_file_path = 'all_benchmarks.json'
    method_name = 'depthanythingv2'
    output_dir = f'{method_name}'
    data_dir = '/mnt/pfs/data/RGBD/moge_eval/'
    
    # load dataset
    datasets = load_datasets_info(json_file_path)
        
    generated_files = []
    for dataset_name, dataset_info in datasets.items():
        dataset_info['path'] = f'{data_dir}/{dataset_name}'
        output_path = generate_yaml_config(method_name, dataset_name, dataset_info, output_dir)
        generated_files.append(output_path)
        print(f"Generate: {output_path}")
    
    print("\Done:")
    for file in generated_files:
        print(f"- {file}")

if __name__ == '__main__':
    main()