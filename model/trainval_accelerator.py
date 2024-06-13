import json
import logging
import math
import os
import random

import datasets
import evaluate
import numpy as np
import torch

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import CONFIG_MAPPING, AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, get_scheduler

from model.nltoolkit import init_nltk, postprocess_text


logger = get_logger(__name__)


def trainval(cfg):
    # Initialize the Natural language toolkit
    init_nltk()
    
    # Initialize the accelerator.
    checkpoint_path = os.path.join(cfg.checkpoint_path, cfg.checkpoint_name)
    accelerator_log_kwargs = {}
    if cfg.use_logger:
        accelerator_log_kwargs["log_with"] = cfg.logger_type
        accelerator_log_kwargs["project_dir"] = checkpoint_path

    accelerator = Accelerator(gradient_accumulation_steps=cfg.gradient_accumulation_steps, **accelerator_log_kwargs)
    
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Reproducibility settings
    if cfg.seed is not None:
        set_seed(cfg.seed)

    if accelerator.is_main_process:
        os.makedirs(checkpoint_path, exist_ok=True)
    accelerator.wait_for_everyone()

    preprocessed_train_dataset_name = f"{cfg.dataset_name}-train-{cfg.obs_len}-{cfg.pred_len}-{cfg.metric}-multimodal.json"
    preprocessed_val_dataset_name = f"{cfg.dataset_name}-val-{cfg.obs_len}-{cfg.pred_len}-{cfg.metric}.json"
    preprocessed_dataset_path = os.path.join(cfg.dataset_path, "preprocessed")

    data_files = {}
    data_files["train"] = os.path.join(preprocessed_dataset_path, preprocessed_train_dataset_name)
    data_files["validation"] = os.path.join(preprocessed_dataset_path, preprocessed_val_dataset_name)

    if not os.path.exists(data_files["train"]) or not os.path.exists(data_files["validation"]):
        raise ValueError(f"Preprocessed dataset files not found: {data_files['train']} or {data_files['validation']}. Please run `./script/preprocessor.sh` first.")
        
    extension = data_files["train"].split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=cfg.cache_dir)

    if cfg.model_config_name or cfg.model_name_or_path:
        config = AutoConfig.from_pretrained(cfg.model_config_name if cfg.model_config_name else cfg.model_name_or_path, trust_remote_code=False, cache_dir=cfg.cache_dir)
    else:
        config = CONFIG_MAPPING[cfg.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if cfg.tokenizer_name or cfg.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name if cfg.tokenizer_name else cfg.model_name_or_path, trust_remote_code=False, cache_dir=cfg.cache_dir, use_fast=not cfg.use_slow_tokenizer)
    else:
        raise ValueError("You are instantiating a new tokenizer from scratch. This is not supported by this script. You can do it from another script, utils/tokenizer, save it, and load it from here, using --tokenizer_name.")

    if cfg.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name_or_path, config=config, trust_remote_code=False, cache_dir=cfg.cache_dir)
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config, trust_remote_code=False)

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    if cfg.tokenizer_name is not None:
        model.resize_token_embeddings(len(tokenizer))

    column_names = raw_datasets["train"].column_names

    history_column = cfg.history_column
    if history_column not in column_names:
        raise ValueError(f"--history_column' value '{cfg.history_column}' needs to be one of: {', '.join(column_names)}")
    future_column = cfg.future_column
    if future_column not in column_names:
        raise ValueError(f"--future_column' value '{cfg.future_column}' needs to be one of: {', '.join(column_names)}")

    padding = "max_length" if cfg.pad_to_max_length else False

    def preprocess_function(examples):
        inputs = examples[history_column]
        targets = examples[future_column]
        model_inputs = tokenizer(inputs, max_length=cfg.max_source_length, padding=padding, truncation=True)
        labels = tokenizer(text_target=targets, max_length=cfg.max_target_length, padding=padding, truncation=True)

        if padding == "max_length":
            labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    with accelerator.main_process_first():
        train_dataset = raw_datasets["train"].map(preprocess_function,
                                                  batched=True,
                                                  num_proc=cfg.preprocessing_num_workers,
                                                  remove_columns=column_names,
                                                  load_from_cache_file=not cfg.overwrite_cache,
                                                  desc="Running tokenizer on train dataset")

        val_dataset = raw_datasets["validation"].map(preprocess_function,
                                                     batched=True,
                                                     num_proc=cfg.preprocessing_num_workers,
                                                     remove_columns=column_names,
                                                     load_from_cache_file=not cfg.overwrite_cache,
                                                     desc="Running tokenizer on val dataset")

    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(tokenizer,
                                           model=model,
                                           label_pad_token_id=label_pad_token_id,
                                           pad_to_multiple_of=8 if accelerator.use_fp16 else None)

    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=cfg.per_device_train_batch_size)
    eval_dataloader = DataLoader(val_dataset, collate_fn=data_collator, batch_size=cfg.per_device_eval_batch_size)

    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [{"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                                     "weight_decay": cfg.weight_decay,},
                                    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                                     "weight_decay": 0.0,},]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=cfg.learning_rate)

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    if cfg.max_train_steps is None:
        cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(name=cfg.lr_scheduler_type,
                                 optimizer=optimizer,
                                 num_warmup_steps=cfg.num_warmup_steps * cfg.gradient_accumulation_steps,
                                 num_training_steps=cfg.max_train_steps * cfg.gradient_accumulation_steps)

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader, lr_scheduler)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    if overrode_max_train_steps:
        cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch
    cfg.num_train_epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)

    checkpointing_steps = cfg.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    if cfg.use_logger:
        experiment_config = cfg
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("Language-Based Trajectory Predictor", experiment_config)

    metric = evaluate.load("rouge")
    total_batch_size = cfg.per_device_train_batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {cfg.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.max_train_steps}")
    
    progress_bar = tqdm(range(cfg.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    
    if cfg.resume_from_checkpoint:
        path = os.path.basename(cfg.resume_from_checkpoint)
        accelerator.print(f"Resumed from checkpoint: {cfg.resume_from_checkpoint}")
        accelerator.load_state(path)

        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            resume_step = int(training_difference.replace("step_", "")) * cfg.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
            completed_steps = resume_step // cfg.gradient_accumulation_stepp

        progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, cfg.num_train_epochs):
        model.train()
        if cfg.use_logger:
            total_loss = 0

        if cfg.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                
                if cfg.use_logger:
                    total_loss += loss.detach().float()
                
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    checkpoint_path_step = os.path.join(checkpoint_path, f"step_{completed_steps}")
                    accelerator.save_state(checkpoint_path_step)

            if completed_steps >= cfg.max_train_steps:
                break

        model.eval()

        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(batch["input_ids"],
                                                                            attention_mask=batch["attention_mask"],
                                                                            max_length=cfg.max_target_length,
                                                                            num_beams=1)
                generated_tokens = accelerator.pad_across_processes(generated_tokens, dim=1, pad_index=tokenizer.pad_token_id)
                labels = batch["labels"]

                if not cfg.pad_to_max_length:
                    labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                generated_tokens, labels = accelerator.gather_for_metrics((generated_tokens, labels))
                generated_tokens = generated_tokens.cpu().numpy()
                labels = labels.cpu().numpy()

                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                generated_tokens = generated_tokens[0] if isinstance(generated_tokens, tuple) else generated_tokens

                if not cfg.use_slow_tokenizer:
                    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                else:
                    filtered_tokens_preds = np.where(generated_tokens >= tokenizer.sp_model.get_piece_size(), 0, generated_tokens)
                    decoded_preds = tokenizer.sp_model.decode(filtered_tokens_preds.tolist())
                    filtered_tokens_labels = np.where(labels >= tokenizer.sp_model.get_piece_size(), 0, labels)
                    decoded_labels = tokenizer.sp_model.decode(filtered_tokens_labels.tolist())
                
                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
                metric.add_batch(predictions=decoded_preds, references=decoded_labels)

        result = metric.compute(use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}

        logger.info(result)

        if cfg.use_logger:
            result["train_loss"] = total_loss.item() / len(train_dataloader)
            result["epoch"] = epoch
            result["step"] = completed_steps
            accelerator.log(result, step=completed_steps)

        if cfg.checkpointing_steps == "epoch":
            checkpoint_path_epoch = os.path.join(checkpoint_path, f"epoch_{epoch}")
            accelerator.save_state(checkpoint_path_epoch)

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(checkpoint_path, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(checkpoint_path)

        all_results = {f"eval_{k}": v for k, v in result.items()}
        with open(os.path.join(checkpoint_path, "all_results.json"), "w") as f:
            json.dump(all_results, f)


if __name__ == "__main__":
    from utils.config import get_exp_config, DotDict
    args = get_exp_config()
    cfg = DotDict(args.__dict__)
    trainval(cfg)
