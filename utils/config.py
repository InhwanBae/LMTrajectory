import os
import json
import argparse


def get_exp_config(file: str):
    r"""Load the configuration files"""

    assert os.path.exists(file), f"File {file} does not exist!"
    file = open(file)
    config = json.load(file)

    def recursive_convert(data):
        if isinstance(data, dict):
            for k, v in data.items():
                data[k] = recursive_convert(v)
            return DotDict(data)
        if isinstance(data, list):
            return [recursive_convert(i) for i in data]
        if isinstance(data, tuple):
            return tuple(recursive_convert(i) for i in data)
        return data

    return recursive_convert(config)


class DotDict(dict):
    r"""dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    __getstate__ = dict
    __setstate__ = dict.update


def print_arguments(args, length=100, sep=': ', delim=' | '):
    r"""Print the arguments in a nice format

    Args:
        args (dict): arguments
        length (int): maximum length of each line
        sep (str): separator between key and value
        delim (str): delimiter between lines
    """

    text = []
    for key in args.keys():
        text.append('{}{}{}'.format(key, sep, args[key]))

    out = ''
    cl = 0
    for n, line in enumerate(text):
        if cl + len(line) > length:
            out += '\n'
            cl = 0
        out += line
        cl += len(line)
        if n != len(text) - 1:
            out += delim
            cl += len(delim)
    print(out)


def parse_args():
    r"""Parse the arguments"""

    from transformers import MODEL_MAPPING, SchedulerType
    MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
    MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

    parser = argparse.ArgumentParser(description="Config")

    # Global config
    parser.add_argument("--config_name", type=str, default="config.json", help="whether to save the config file")
    parser.add_argument("--train", action="store_true", help="whether to train the model")
    parser.add_argument("--eval", action="store_true", help="whether to evaluate the model")

    # Model config
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--model_config_name", type=str, default=None, help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--model_type", type=str, default=None, help="Model type to use if training from scratch.", choices=MODEL_TYPES)    

    # Dataset config
    parser.add_argument("--dataset_path", type=str, default="./datasets/", help="Path to the dataset.")
    parser.add_argument("--dataset_name", type=str, default=None, help="The name of the dataset to use.")
    parser.add_argument("--metric", type=str, default="pixel", help="The metric to use for evaluation.", choices=["meter", "pixel"])
    parser.add_argument("--obs_len", type=int, default=8, help="The length of the observed trajectory.")
    parser.add_argument("--pred_len", type=int, default=12, help="The length of the predicted trajectory.")
    parser.add_argument("--preprocessing_num_workers", type=int, default=None, help="The number of processes to use for the preprocessing.")
    parser.add_argument("--cache_dir", type=str, default="./.cache/", help="Where to store the pretrained models downloaded from huggingface.co")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets")
    
    # Tokenizer config
    parser.add_argument("--tokenizer_name", type=str, default=None, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--use_slow_tokenizer", action="store_true", help="If passed, will use a slow tokenizer (not backed by the huggingface Tokenizers library).")
    parser.add_argument("--pad_to_max_length", action="store_true", help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.")
    parser.add_argument("--history_column", type=str, default="observation", help="The name of the column in the datasets containing the full texts (for summarization).")
    parser.add_argument("--future_column", type=str, default="forecast", help="The name of the column in the datasets containing the summaries (for summarization).")
    parser.add_argument("--max_source_length", type=int, default=512, help="The maximum total input sequence length after tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", type=int, default=128, help="The maximum total sequence length for target text after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.during ``evaluate`` and ``predict``.")
    
    # Trainer config
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear", help="The scheduler type to use.", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")

    # Output config
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoint/", help="Where to store the final model.")
    parser.add_argument("--checkpoint_name", type=str, default=None, help="The name of the checkpoint.")
    parser.add_argument("--checkpointing_steps", type=str, default=None, help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="If the training should continue from a checkpoint folder.")
    parser.add_argument("--use_logger", action="store_true", help="Whether to enable experiment trackers for logging.")
    parser.add_argument("--logger_type", type=str, default="", help='The integration to report the results and logs to. Supported platforms are `"tensorboard"`,`"wandb"`,  `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. Only applicable when `--use_logger` is passed.')
    
    # Test config
    parser.add_argument("--num_beams", type=int, default=3, help="Number of beams to use for evaluation. This argument will be passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic evaluation process")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling. This arguement will be used for stochastic sampling.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature value. This argument will be used for stochastic sampling.")
    parser.add_argument("--best_of_n", type=int, default=20, help="Best of n sampling. This argument will be used for stochastic sampling.")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to generate. This argument will be used for stochastic sampling.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # Print arguments
    print_arguments(args.__dict__)

    # Save arguments
    config_path = './config/'
    config_file = args.config_name
    os.makedirs(config_path, exist_ok=True)
    with open(config_path + config_file, 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    # Load arguments
    cfg = get_exp_config(config_path + config_file)
    print_arguments(cfg)
