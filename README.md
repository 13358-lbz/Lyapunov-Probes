# Lyapunov Probes

Official repository for **Lyapunov Probes for Hallucination Detection in Large Foundation Models**.

🚀 **Accepted to CVPR 2026**

## Overview

**Lyapunov Probes** is a lightweight hallucination detection framework for Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs).  
We formulate hallucination detection from the perspective of dynamical systems and Lyapunov stability theory, modeling hallucinations as unstable regions near knowledge boundaries in the representation space.

## Paper

**Lyapunov Probes for Hallucination Detection in Large Foundation Models**

📄 Paper: https://arxiv.org/pdf/2603.06081



## Getting Started

### Installation
The project uses `poetry` for dependency management and packaging. The latest version and instructions can be
found on [https://python-poetry.org](https://python-poetry.org/docs/).
official installer:
```shell
curl -sSL https://install.python-poetry.org | python3 -
```

```shell
poetry install
```

>Using poetry takes care of all dependencies, and therefore removes the need for requirements.txt. Should you still need that file for any reason, it can be generated using:
```shell
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

#### Accelerate
This project uses huggingface's accelerate for GPU management. 
Feel free to launch accelerate config to get the most out of it.

## Usage
### data generation pipeline:
Data has at least the following columns: ["text","uuid","is_factual"]. If the paraphrasing option is used, a ["paraphrase"] column will be used.

To prepare the True/False Lama TRex dataset use dataset_prep.py, which will create a test and train set in a data folder at root.
To experiment with the PopQA dataset :
 - Download csv file from the following [link](https://github.com/AlexTMallen/adaptive-retrieval/blob/main/data/popQA.tsv) (tested on 25/06/2024)
 - run slot_filling.py to get a specific model's ability to correctly answer each question, and generate the ["is_factual"] column

### to run experiments:
1. run training pipeline ("hidden") method
2. run main.py (all results are saved except for consistency)
3. run consistency pipeline
example scripts: scripts/main.sh, scripts/main_pop.sh, scripts/main_translated.sh, scripts/main_pik_lama.sh


### training pipeline - run, in order:
example script: scripts/extract_hidden.sh 
1. evaluation/extract_hidden_layers.py (runs a given model on a given dataset, and saves the hidden dimensions + labels for training)
2. train_scorer_2 (takes as input the hidden dimensions from previous script, runs gradient descent, saves the resulting model)

## others

The codebase is based on https://github.com/amazon-science/factual-confidence-of-llms . We sincerely thank the authors for their excellent work and for making their code publicly available.

For more information, please contact luanbz0075@gmail.com.

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{luan2026lyapunov,
  title={Lyapunov Probes for Hallucination Detection in Large Foundation Models},
  author={Luan, Bozhi and Li, Gen and Qin, Yalan and Guo, Jifeng and Zhou, Yun and Wu, Faguo and Zheng, Hongwei and Wu, Wenjun and Fan, Zhaoxin},
  journal={arXiv preprint arXiv:2603.06081},
  year={2026}
}
