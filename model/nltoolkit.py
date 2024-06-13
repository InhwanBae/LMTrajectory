import nltk
from filelock import FileLock
from transformers.utils import is_offline_mode


def init_nltk():
    r"""Initialize nltk data files"""
    
    try:
        nltk.data.find("tokenizers/punkt")
    except (LookupError, OSError):
        if is_offline_mode():
            raise LookupError("Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files")
        with FileLock(".lock") as lock:
            nltk.download("punkt", quiet=True)


def preprocess_function(examples, cfg, history_column, future_column, tokenizer, padding="max_length"):
        inputs = examples[history_column]
        targets = examples[future_column]
        model_inputs = tokenizer(inputs, max_length=cfg.max_source_length, padding=padding, truncation=True)
        labels = tokenizer(text_target=targets, max_length=cfg.max_target_length, padding=padding, truncation=True)

        if padding == "max_length":
            labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rouge score expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels
