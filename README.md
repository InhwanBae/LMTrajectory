<!--<h2 align="center">Can Language Beat Numerical Regression?<br>Language-Based Multimodal Trajectory Prediction</h2>-->
<!--<h2 align="center">Social Reasoning-Aware Trajectory Prediction<br>via Multimodal Language Model</h2>-->
<h2 align="center">
  Can $\large{\color{Orange}{\textbf{\textsf{Language}}}}$ Beat $\large{\color{MidnightBlue}{\textbf{\textsf{Numerical Regression}}}}$?<br>Language-Based Multimodal Trajectory Prediction
  <br>$\tiny{—~and~—}$
  <br>Social Reasoning-Aware Trajectory Prediction<br>via $\large{\color{Maroon}{\textbf{\textsf{Multimodal Language Model}}}}$
</h2>
<p align="center">
  <a href="https://InhwanBae.github.io/"><strong>Inhwan Bae</strong></a>
  ·  
  <a href="https://leejunoh.com/"><strong>Junoh Lee</strong></a>
  ·
  <a href="https://scholar.google.com/citations?user=Ei00xroAAAAJ"><strong>Hae-Gon Jeon</strong></a>
  <br>
  CVPR 2024  &  TPAMI
</p>

<p align="center">
  <a href="https://inhwanbae.github.io/publication/lmtrajectory/"><strong><code>Project Page</code></strong></a>
  <a href="https://arxiv.org/abs/2403.18447"><strong><code>CVPR Paper</code></strong></a>
  <a href="https://ieeexplore.ieee.org/abstract/document/11045841"><strong><code>TPAMI Paper</code></strong></a>
  <a href="https://github.com/InhwanBae/LMTrajectory"><strong><code>Source Code</code></strong></a>
  <a href="#-citation"><strong><code>Related Works</code></strong></a>
</p>

<div align='center'>
  <br><img src="img/lmtraj-model.gif" width=70%>
  <br>Traditional vs. Our language-based trajectory prediction, LMTraj.
</div>

<!--<br>This repository contains the code for the LMTrajectory framework.-->
<br>**Summary**: **Language model**-based, **Multimodal input**, **Multimodal output**, **Multi-task training** approach for **Zero-shot** and **Supervised** human trajectory prediction. 

<br>

## 💬 LMTrajectory Framework 🗨️
* **Prompt-Based Approach**: Moving away from conventional numerical regression models, we reframe the task into a prompt-based question-answering perspective.
* **Social Reasoning**: Beyond physics-based mathematical interaction modeling, our approach leverages language models to incorporate social reasoning.
* **Multi-Task Training**: Supplementary tasks enhance the model's ability to grasp higher-level context through multi-task training.
* **Numerical Tokenizer**: Our numerical tokenizer effectively separates text and numbers, enabling the model to learn correlations in sequential data.
* **SOTA Performance**: Our holistic solution achieves state-of-the-art results on trajectory prediction benchmarks traditionally dominated by numerical regressors.

<br>

## ❄️ Zero-Shot Evaluation ❄️
### Setup
**Environment**
<br>All models were tested on Ubuntu 20.04 with Python 3.10 and PyTorch 2.0.1 with CUDA 11.7.
Dependencies include Python packages such as `scipy`, `simdkalman` and `openai==0.28.0`.

**Dataset**
<br>Preprocessed [ETH](https://data.vision.ee.ethz.ch/cvl/aem/ewap_dataset_full.tgz) and [UCY](https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data) datasets are released in this repository. 
The train/validation/test splits are the same as those fond in [Social-GAN](https://github.com/agrimgupta92/sgan).

**Sample**
<br>We provide our zero-shot prediction results in the [release section](https://github.com/InhwanBae/LMTrajectory/releases/download/v1.0/LMTraj-ZERO_output_trajectory.zip). 
These results include all multimodal trajectories and are available for use in future zero-shot research.

### Evaluate LMTraj-ZERO
**Preliminary**
<br>To evaluate our LMTraj-ZERO model, you will need an `OPENAI_API_KEY` to access the OpenAI API. Create the API key using the [instruction](https://platform.openai.com/docs/api-reference/authentication) provided by OpenAI, and then paste the key into `./zero-shot/chatgpt_trajectory_predictor_v3.py` [line 25](https://github.com/InhwanBae/LMTrajectory/blob/main/zero-shot/chatgpt_trajectory_predictor_v3.py#L25).

**Prediction**
<br>We provide scripts to evaluate our LMTraj-ZERO model for all datasets simultaneously.
Two scripts are provided in `./zero-shot/chatgpt_sequential_v3.sh` and `./zero-shot/chatgpt_multi_v3.sh`. The former script is used to evaluate our model step-by-step, and the latter script is used to evaluate our model with a thread pool for faster inference.
```bash
# Choose one of the following scripts to evaluate our LMTraj-ZERO model.
./chatgpt_sequential_v3.sh -d <DATASET_ID> -m <LLM_MODEL_ID>
./chatgpt_multi_v3.sh -d <DATASET_ID> -m <LLM_MODEL_ID>

# Supported dataset id: 0 (ETH), 1 (HOTEL), 2 (UNIV), 3 (ZARA1), 4 (ZARA2)
# Supported llm model id: 0 (gpt-3.5-turbo-0301), 1 (gpt-4-0314), 2 (gpt-3.5-turbo-1106), 3 (gpt-4-1106-preview)

# Examples
cd zero-shot
./chatgpt_multi_v3.sh -d 0 -m 3
./chatgpt_multi_v3.sh -d 1 -m 3
```
If an error is encountered, your progress will be saved. When you rerun the same script, it will skip the parts that were successfully executed and only regenerate the paths where issues occurred.

If you want to run the model with custom hyperparameters or other models available by [OpenAI](https://platform.openai.com/docs/models), use `./zero-shot/chatgpt_trajectory_predictor_v3.py` instead of the script file. 
<br>*Warning: A misclick could upgrade you to OpenAI Tier 5, as it did for me :(* 

**Evaluation**
<br>As the final step, we provide code to evaluate the trajectories generated by our LMTraj-ZERO. To evaluate, first combine the predicted trajectories into a single JSON file.
```bash
python ./zero-shot/chatgpt-fragmented_dump_combiner.py --dataset <DATASET_ID> --model <LLM_MODEL_ID>

# Supported dataset id: 0 (ETH), 1 (HOTEL), 2 (UNIV), 3 (ZARA1), 4 (ZARA2)
# Supported llm model id: 0 (gpt-3.5-turbo-0301), 1 (gpt-4-0314), 2 (gpt-3.5-turbo-1106), 3 (gpt-4-1106-preview)

# Examples
python ./zero-shot/chatgpt-fragmented_dump_combiner.py --dataset 0 --model 3
python ./zero-shot/chatgpt-fragmented_dump_combiner.py --dataset 1 --model 3
```

Next, evaluate the combined trajectories using ADE and FDE metrics.
```bash
python ./zero-shot/compute_ade_fde_from_dump.py --dataset <DATASET_ID> --model <LLM_MODEL_ID>

# Supported dataset id: 0 (ETH), 1 (HOTEL), 2 (UNIV), 3 (ZARA1), 4 (ZARA2)
# Supported llm model id: 0 (gpt-3.5-turbo-0301), 1 (gpt-4-0314), 2 (gpt-3.5-turbo-1106), 3 (gpt-4-1106-preview)

# Examples
python ./zero-shot/compute_ade_fde_from_dump.py --dataset 0 --model 3
python ./zero-shot/compute_ade_fde_from_dump.py --dataset 1 --model 3
```

**Results**
<table><thead><tr><th rowspan="2"><sub><b>LMTraj-ZERO</b></sub></th><th colspan="2"><sub><b>ETH</b></sub></th><th colspan="2"><sub><b>HOTEL</b></sub></th><th colspan="2"><sub><b>UNIV</b></sub></th><th colspan="2"><sub><b>ZARA1</b></sub></th><th colspan="2"><sub><b>ZARA2</b></sub></th><th colspan="2"><sub><b>AVG</b></sub></th></tr>
<tr><th><sub><b>ADE</b></sub></th><th><sub><b>FDE</b></sub></th><th><sub><b>ADE</b></sub></th><th><sub><b>FDE</b></sub></th><th><sub><b>ADE</b></sub></th><th><sub><b>FDE</b></sub></th><th><sub><b>ADE</b></sub></th><th><sub><b>FDE</b></sub></th><th><sub><b>ADE</b></sub></th><th><sub><b>FDE</b></sub></th><th><sub><b>ADE</b></sub></th><th><sub><b>FDE</b></sub></th></tr></thead><tbody>
<tr><td><sub><b>gpt-3.5-turbo-0301</b></sub></td><td><sub>1.0668</sub></td><td><sub>1.8241</sub></td><td><sub>0.4229</sub></td><td><sub>0.6538</sub></td><td><sub>0.5570</sub></td><td><sub>0.9836</sub></td><td><sub>0.4715</sub></td><td><sub>0.9073</sub></td><td><sub>0.3878</sub></td><td><sub>0.7056</sub></td><td><sub>0.5812</sub></td><td><sub>1.0149</sub></td></tr>
<tr><td><sub><b>gpt-3.5-turbo-1106</b></sub></td><td><sub></sub></td><td><sub></sub></td><td><sub>0.4713</sub></td><td><sub>0.6297</sub></td><td><sub></sub></td><td><sub></sub></td><td><sub></sub></td><td><sub></sub></td><td><sub></sub></td><td><sub></sub></td><td><sub></sub></td><td><sub></sub></td></tr>
<tr><td><sub><b>gpt-4-0314</b></sub></td><td><sub>0.7978</sub></td><td><sub>1.6446</sub></td><td><sub>0.2001</sub></td><td><sub>0.3658</sub></td><td><sub>0.3709</sub></td><td><sub>0.7675</sub></td><td><sub>0.3268</sub></td><td><sub>0.6638</sub></td><td><sub>0.2386</sub></td><td><sub>0.4998</sub></td><td><sub>0.3868</sub></td><td><sub>0.7883</sub></td></tr>
<tr><td><sub><b>gpt-4-1106-preview</b></sub></td><td><sub></sub></td><td><sub></sub></td><td><sub>0.1757</sub></td><td><sub>0.3279</sub></td><td><sub></sub></td><td><sub></sub></td><td><sub></sub></td><td><sub></sub></td><td><sub></sub></td><td><sub></sub></td><td><sub></sub></td><td><sub></sub></td></tr></tbody></table>

### Evaluate Algorithmic Models
We provide four algorithmic models for comparison in zero-shot trajectory prediction task, available in `./zero-shot/algorithmic_model_benchmark.py`.
The source code supports four extrapolation methods: stop, linear extrapolation, cubic extrapolation and Kalman filter.
```bash
python ./zero-shot/algorithmic_model_benchmark.py --model <MODEL_TYPE>

# Examples
python ./zero-shot/algorithmic_model_benchmark.py --model stop
python ./zero-shot/algorithmic_model_benchmark.py --model linear
python ./zero-shot/algorithmic_model_benchmark.py --model cubic
python ./zero-shot/algorithmic_model_benchmark.py --model kalman
```

<br>

## 🔥 Supervised Training & Evaluation 🔥
### Setup
**Environment**
<br>All models were tested on Ubuntu 20.04 with Python 3.10 and PyTorch 2.0.1 with CUDA 11.7.
Dependencies include Python packages such as `transformers`, `accelerate`, `nltk` and `sentencepiece`.

**Dataset**
<br>Preprocessed [ETH](https://data.vision.ee.ethz.ch/cvl/aem/ewap_dataset_full.tgz) and [UCY](https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data) datasets are released in this repository. 
The train/validation/test splits are the same as those fond in [Social-GAN](https://github.com/agrimgupta92/sgan).

### Preliminary
We provide [preprocessed datasets](https://github.com/InhwanBae/LMTrajectory/releases/tag/v0.1), [pretrained tokenizers](https://github.com/InhwanBae/LMTrajectory/releases/tag/v1.0), [and models](https://github.com/InhwanBae/LMTrajectory/releases/tag/v1.0) for training and evaluation. Download these files and extract them into the root folder of the project. This will allow you to skip preprocessing and evaluate our LMTraj-SUP model immediately.

Additionally, we provide instructions for preprocessing and training the data yourself. Follow these steps:

**Dataset Preprocessing**
<br>To maximize GPU utilization and reduce training time, we preprocess the training data. First, generate text descriptions of the dataset environment using the image captioning model located at `./model/imagemodel.py`. This script automatically loads the pretrained model and saves the captions in the `./datasets/image/` folder.
```bash
python ./model/imagemodel.py
```

Next, to preprocess all datasets simultaneously, run the `./script/preprocessor.sh` script. This process takes about 2 hours and generates preprocessed JSON files in the `./datasets/preprocessed/` folder.
```bash
./script/preprocessor.sh
```

If you prefer to preprocess the datasets individually, use `./utils/preprocessor.py` instead of the script.
```bash
python ./utils/preprocessor.py --dataset <DATASET_NAME> --phase <TRAINING_PHASE>

# Supported dataset name: eth, hotel, univ, zara1, zara2
# Supported training phase: train, val, test

# Examples
python ./utils/preprocessor.py --dataset eth --phase train
python ./utils/preprocessor.py --dataset hotel --phase val
python ./utils/preprocessor.py --dataset univ --phase test
```

**Tokenizer Training**
<br>Next, train the tokenizer to optimize it for numerical data. You can train the tokenizer yourself using `/utils/tokenizer.py`. This process requires a system with more than 2TB of RAM and takes approximately 12 hours for each.
```bash
python ./utils/tokenizer.py --dataset <DATASET_NAME> --model <TOKENIZER_MODEL> --metric <PIXEL_OR_METER>

# Supported dataset name: eth, hotel, univ, zara1, zara2
# Supported tokenizer model type: char, word, unigram, bpe
# Supported metric type: pixel, meter

# Examples
python ./utils/tokenizer.py --dataset eth --model bpe --metric pixel
```

### Train LMTraj-SUP
To train our LMTrajectory model, you will use `./trainval.py`. We leverage the `accelerate` library to maximize training efficiency. First, configure your system by running `accelerate config` in the shell. You can find detailed instructions in the [Accelerate documentation](https://huggingface.co/docs/accelerate/basic_tutorials/install).

To train the model, use the following command:
```shell
accelerate launch trainval.py \
    --cfg ./config/config-pixel.json \
    --dataset eth \
    --tag LMTraj-SUP-eth
```

If you want to train the LMTraj-SUP model on both the ETH and UCY datasets simultaneously, we provide a bash script:
```bash
./script/trainval_all.sh
```

The training process uses 8x NVIDIA RTX 4090 GPUs at 100% utilization and takes approximately 2 to 4 hours. After training, select the best weight file from the checkpoint epochs.

### Evaluate LMTraj-SUP
Finally, to evaluate our LMTrajectory model, use `./trainval.py` again with the `--test` tag. This will perform the evaluation. You can conduct both stochastic and deterministic trajectory predictions using a single pretrained weight file.

For stochastic trajectory prediction, use:
```bash
accelerate launch trainval.py \
    --cfg ./config/config-pixel.json \
    --dataset eth \
    --tag LMTraj-SUP-eth \ 
    --test
```
For deterministic trajectory prediction, use:
```bash
accelerate launch trainval.py \
    --cfg ./config/config-pixel-deterministic.json \
    --dataset eth \
    --tag LMTraj-SUP-eth \ 
    --test
```

To evaluate our LMTraj-SUP model on both the ETH and UCY datasets simultaneously, we provide the following bash scripts for a simplified execution:
```bash
./script/eval_all.sh
./script/eval_all_deterministic.sh
```

**Results**
<table><thead><tr><th rowspan="2"><sub><b>LMTraj-SUP</b></sub></th><th colspan="2"><sub><b>ETH</b></sub></th><th colspan="2"><sub><b>HOTEL</b></sub></th><th colspan="2"><sub><b>UNIV</b></sub></th><th colspan="2"><sub><b>ZARA1</b></sub></th><th colspan="2"><sub><b>ZARA2</b></sub></th><th colspan="2"><sub><b>AVG</b></sub></th></tr>
<tr><th><sub><b>ADE</b></sub></th><th><sub><b>FDE</b></sub></th><th><sub><b>ADE</b></sub></th><th><sub><b>FDE</b></sub></th><th><sub><b>ADE</b></sub></th><th><sub><b>FDE</b></sub></th><th><sub><b>ADE</b></sub></th><th><sub><b>FDE</b></sub></th><th><sub><b>ADE</b></sub></th><th><sub><b>FDE</b></sub></th><th><sub><b>ADE</b></sub></th><th><sub><b>FDE</b></sub></th></tr></thead><tbody>
<tr><td><sub><b>Deterministic w/ image</b></sub></td><td><sub>0.6549</sub></td><td><sub>1.0377</sub></td><td><sub>0.2640</sub></td><td><sub>0.4583</sub></td><td><sub>0.5715</sub></td><td><sub>1.1579</sub></td><td><sub>0.5119</sub></td><td><sub>1.0066</sub></td><td><sub>0.3802</sub></td><td><sub>0.7408</sub></td><td><sub>0.4765</sub></td><td><sub>0.8803</sub></td></tr>
<tr><td><sub><b>Deterministic w/o image</b></sub></td><td><sub>0.6724</sub></td><td><sub>1.2388</sub></td><td><sub>0.2498</sub></td><td><sub>0.4331</sub></td><td><sub>0.5723</sub></td><td><sub>1.1612</sub></td><td><sub>0.5090</sub></td><td><sub>1.0018</sub></td><td><sub>0.3827</sub></td><td><sub>0.7471</sub></td><td><sub>0.4772</sub></td><td><sub>0.9164</sub></td></tr>
<tr><td><sub><b>Stochastic w/ image</b></sub></td><td><sub>0.4087</sub></td><td><sub>0.5011</sub></td><td><sub>0.1200</sub></td><td><sub>0.1558</sub></td><td><sub>0.2178</sub></td><td><sub>0.3440</sub></td><td><sub>0.1992</sub></td><td><sub>0.3183</sub></td><td><sub>0.1748</sub></td><td><sub>0.2720</sub></td><td><sub>0.2241</sub></td><td><sub>0.3182</sub></td></tr>
<tr><td><sub><b>Stochastic w/o image</b></sub></td><td><sub>0.4106</sub></td><td><sub>0.6188</sub></td><td><sub>0.1212</sub></td><td><sub>0.1595</sub></td><td><sub>0.2188</sub></td><td><sub>0.3465</sub></td><td><sub>0.2018</sub></td><td><sub>0.3225</sub></td><td><sub>0.1756</sub></td><td><sub>0.2760</sub></td><td><sub>0.2256</sub></td><td><sub>0.3447</sub></td></tr></tbody></table>

<br>

## 📖 Citation
If you find this code useful for your research, please cite our trajectory prediction papers :)

[**`🏢🚶‍♂️ CrowdES (CVPR'25) 🏃‍♀️🏠`**](https://github.com/InhwanBae/Crowd-Behavior-Generation) **|**
[**`💭 VLMTrajectory (TPAMI) 💭`**](https://github.com/InhwanBae/LMTrajectory) **|**
[**`💬 LMTrajectory (CVPR'24) 🗨️`**](https://github.com/InhwanBae/LMTrajectory) **|**
[**`1️⃣ SingularTrajectory (CVPR'24) 1️⃣`**](https://github.com/InhwanBae/SingularTrajectory) **|**
[**`🌌 EigenTrajectory (ICCV'23) 🌌`**](https://github.com/InhwanBae/EigenTrajectory) **|** 
[**`🚩 Graph‑TERN (AAAI'23) 🚩`**](https://github.com/InhwanBae/GraphTERN) **|**
[**`🧑‍🤝‍🧑 GP‑Graph (ECCV'22) 🧑‍🤝‍🧑`**](https://github.com/InhwanBae/GPGraph) **|**
[**`🎲 NPSN (CVPR'22) 🎲`**](https://github.com/InhwanBae/NPSN) **|**
[**`🧶 DMRGCN (AAAI'21) 🧶`**](https://github.com/InhwanBae/DMRGCN)

```bibtex
@inproceedings{bae2024lmtrajectory,
  title={Can Language Beat Numerical Regression? Language-Based Multimodal Trajectory Prediction},
  author={Bae, Inhwan and Lee, Junoh and Jeon, Hae-Gon},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}

@article{bae2025vlmtrajectory,
  title={Social Reasoning-Aware Trajectory Prediction via Multimodal Language Model},
  author={Bae, Inhwan and Lee, Junoh and Jeon, Hae-Gon},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025}
}
```
<details open>
  <summary>More Information (Click to expand)</summary>

```bibtex
@inproceedings{bae2025crowdes,
  title={Continuous Locomotive Crowd Behavior Generation},
  author={Bae, Inhwan and Lee, Junoh and Jeon, Hae-Gon},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}

@inproceedings{bae2024singulartrajectory,
  title={SingularTrajectory: Universal Trajectory Predictor Using Diffusion Model},
  author={Bae, Inhwan and Park, Young-Jae and Jeon, Hae-Gon},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}

@inproceedings{bae2023eigentrajectory,
  title={EigenTrajectory: Low-Rank Descriptors for Multi-Modal Trajectory Forecasting},
  author={Bae, Inhwan and Oh, Jean and Jeon, Hae-Gon},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}

@article{bae2023graphtern,
  title={A Set of Control Points Conditioned Pedestrian Trajectory Prediction},
  author={Bae, Inhwan and Jeon, Hae-Gon},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}

@inproceedings{bae2022gpgraph,
  title={Learning Pedestrian Group Representations for Multi-modal Trajectory Prediction},
  author={Bae, Inhwan and Park, Jin-Hwi and Jeon, Hae-Gon},
  booktitle={Proceedings of the European Conference on Computer Vision},
  year={2022}
}

@inproceedings{bae2022npsn,
  title={Non-Probability Sampling Network for Stochastic Human Trajectory Prediction},
  author={Bae, Inhwan and Park, Jin-Hwi and Jeon, Hae-Gon},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}

@article{bae2021dmrgcn,
  title={Disentangled Multi-Relational Graph Convolutional Network for Pedestrian Trajectory Prediction},
  author={Bae, Inhwan and Jeon, Hae-Gon},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2021}
}
```
</details>

<br>
