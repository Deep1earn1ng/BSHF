import os
from yacs.config import CfgNode as CN

def load_config(config_file):
    cfg = CN(new_allowed=True)
    if os.path.exists(config_file):
        cfg.merge_from_file(config_file)
    else:
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    return cfg

def check_cfg_keys(cfg):
    pass

def save_config(cfg, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "config.yml")
    with open(save_path, 'w') as f:
        f.write(cfg.dump())