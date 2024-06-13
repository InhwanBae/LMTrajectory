import os
import sentencepiece as spm
import json
from tqdm import tqdm
from transformers import T5Tokenizer
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="eth", type=str, help="Dataset name to train tokenizer ('eth', 'hotel', 'univ', 'zara1', 'zara2').")
parser.add_argument('--model', default="bpe", type=str, help="Tokenizer model type ('char', 'word', 'unigram', 'bpe').")
parser.add_argument('--metric', default="pixel", type=str, help="metric type ('pixel', 'meter').")
args = parser.parse_args()

tokenizer_basedir = "./checkpoint/tokenizer/"
os.makedirs(tokenizer_basedir, exist_ok=True)

dataset = args.dataset
modeltype = args.model
metric = args.metric

filename = f"trajectoryspiece-{metric}-{modeltype}"

# Export raw json data to text file
if not os.path.exists(tokenizer_basedir + f"{dataset}-all_data-8-12-{metric}-multimodal.txt"):
    with open(f"./datasets/preprocessed/{dataset}-train-8-12-{metric}-multimodal.json") as json_file:
        with open(tokenizer_basedir + f"{dataset}-data-8-12-{metric}-multimodal.txt", 'w', encoding='utf-8') as f:
            for line in tqdm(json_file):
                json_data = json.loads(line)
                f.write(json_data["observation"] + '\n')
                f.write(json_data["forecast"] + '\n')

spm.SentencePieceTrainer.train(
    input=tokenizer_basedir + f"{dataset}-data-8-12-{metric}-multimodal.txt",
    model_prefix=tokenizer_basedir + filename,
    vocab_size=1224,
    unk_id=3,
    bos_id=1,
    eos_id=2,
    pad_id=0,
    control_symbols="[PAD],[UNK],[CLS],[SEP],[MASK]",
    model_type=modeltype,  
    train_extremely_large_corpus=True,
    # use_all_vocab=True,
    character_coverage=1.0,  # 0.99995
)

vocab_file = tokenizer_basedir + filename + ".model"
sp_model = spm.SentencePieceProcessor()
sp_model.Load(vocab_file)

print("vocab size:", sp_model.vocab_size())

from sentencepiece import sentencepiece_model_pb2
m = sentencepiece_model_pb2.ModelProto()
with open(vocab_file, 'rb') as f:
    m.ParseFromString(f.read())

with open(tokenizer_basedir + filename + ".txt", 'w', encoding='utf-8') as f:
    f.write("# trainer_spec\n")
    f.write(m.trainer_spec.__repr__())
    m.normalizer_spec.precompiled_charsmap = b''
    f.write("# normalizer_spec\n")
    f.write(m.normalizer_spec.__repr__())
    f.write("# pieces\n")
    for piece in m.pieces:
        f.write(piece.piece + '\n')

vocab_file = tokenizer_basedir + filename + ".model"
tokenizer = T5Tokenizer(vocab_file=vocab_file)
tokenizer.save_pretrained(tokenizer_basedir + filename + '/')
