# PATCH

**Authors**: Yuying Shang*, Xinyi Zeng*, Yutao Zhu*, Xiao Yang, Zhengwei Fang, Jingyuan Zhang, Jiawei Chen, Zinan Liu, Yu Tian†

This is the official repository for **"From Pixels to Tokens: Revisiting Object Hallucinations in Large Vision-Language Models"** (Accepted at ACM MM 2025)

---

## 🚀 Training PATCH (MiniGPTv2 with POPE)

### 1. Environment Setup
MiniGPT-v2 is based on LLaMA2-Chat-7B. For details about model parameters and downloading instructions, please refer to the [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4).
```bash
conda env create -f environment.yml
conda activate minigptv
```

### 2. Configure Training Parameters

Modify the following configuration files as needed:
```
./MiniGPT4/minigpt4/configs/datasets/pope/default.yaml
./MiniGPT4/minigpt4/configs/models/minigpt_v2.yaml
./MiniGPT4/train_configs/minigptv2_fintune.yaml
./MiniGPT4/scripts/train.sh
```

### 3. Run Training
```bash
bash ./MiniGPT4/scripts/train.sh
```
To run training in multiple loops:
```bash
bash ./MiniGPT4/scripts/train_loop.sh
```

## 📊 Evaluation (MiniGPTv2 with POPE)

### 1. Download Pretrained Weights

Please download the best virtual token weights from the provided link: 
[PATCH_ckpt](https://drive.google.com/file/d/1_SLslW5MlXKfUiEfNeSkLKf0YA2-eB87/view?usp=sharing)

### 2. Configure Evaluation Parameters

Modify the following configuration files as needed:
```
- ./MiniGPT4/eval_configs/minigptv2_pope.yaml
- ./MiniGPT4/scripts/eval.sh
```

### 3. Run Evaluation
```bash
bash ./MiniGPT4/scripts/eval.sh
```

## Acknowledgement
- [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4): We greatly thank this amazing work as this repository is built upon its MiniGPT-v2 architecture.
- [SPRING](https://github.com/DaoD/SPRING): We thank this work for its inspiration.

Please kindly cite using this BibTeX if our work helps your research:
```bash
@article{shang2024pixels,
  title={From pixels to tokens: Revisiting object hallucinations in large vision-language models},
  author={Shang, Yuying and Zeng, Xinyi and Zhu, Yutao and Yang, Xiao and Fang, Zhengwei and Zhang, Jingyuan and Chen, Jiawei and Liu, Zinan and Tian, Yu},
  journal={arXiv preprint arXiv:2410.06795},
  year={2024}
}
```

