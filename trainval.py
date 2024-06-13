import argparse
from utils.config import get_exp_config, print_arguments

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default="./config/config.json", type=str, help="Config file path.")
    parser.add_argument('--dataset', default="eth", type=str, help="Dataset name.", choices=["eth", "hotel", "univ", "zara1", "zara2"])
    parser.add_argument('--tag', default="LMTraj", type=str, help="Personal tag for the model.")
    parser.add_argument('--test', default=False, action='store_true', help="Evaluation mode.")

    args = parser.parse_args()

    print("===== Arguments =====")
    print_arguments(vars(args))

    print("===== Configs =====")
    cfg = get_exp_config(args.cfg)
    print_arguments(cfg)

    # Update configs
    cfg.dataset_name = args.dataset
    cfg.checkpoint_name = args.tag

    if not args.test:
        # Training phase
        from model.trainval_accelerator import *
        trainval(cfg)

    else:
        # Evaluation phase
        from model.eval_accelerator import *
        test(cfg)
