# Bridging Regional and Global Languages: Handwritten Gujarati Text Recognition and Translation Using Deep Neural Network

This repository contains the implementation for a handwritten Gujarati text recognition and translation pipeline. The system processes Gujarati handwritten text images through preprocessing, segmentation, ResNet-50 feature extraction, Transformer-based sequence modeling, autoregressive decoding with beam search, and translation into Hindi and English.

## Authors

**Mehulkumar Dalwadi**  
Faculty of IT and Computer Science, Parul University, Waghodia, Vadodara, Gujarat-391760, India  
Email: mehulkumardalwadi.phd@gmail.com  
ORCID: https://orcid.org/0009-0009-7878-0995

**Dr. Abhishek Mehta**  
Department of MCA, Parul Institute of Computer Application, Faculty of IT and Computer Science, Parul University, Waghodia, Vadodara, Gujarat-391760, India  


## Project Overview

The objective of this work is to recognize handwritten Gujarati text and translate the recognized output into regional and global languages. The implementation follows an end-to-end workflow:

1. Data collection from handwritten Gujarati samples and text sources.
2. Image preprocessing using binarization, noise removal, and skew correction.
3. Line-level segmentation of handwritten text.
4. Feature extraction using a ResNet-50 backbone.
5. Token preparation and train/validation/test splitting.
6. Transformer-based sequence modeling.
7. Autoregressive decoding using beam search.
8. Evaluation using character error rate (CER), word error rate (WER), and other recognition metrics.
9. Gujarati-to-Hindi and Gujarati-to-English translation through a GUI module.

## Repository Structure

```text
.
├── step1_data_collection.py
├── step2_binarization.py
├── step3_noise_removal.py
├── step4_deskew.py
├── step5_line_segmentation.py
├── step6_line_feature_extraction_split.py
├── step7.py
├── step8.py
├── step9.py
├── TRANSLATION_MODULE/
│   ├── translation_module.py
│   └── translation_gui.py
├── requirements.txt
├── .gitignore
└── README.md
```

Generated folders such as `STEP*_OUTPUTS`, trained model checkpoints, virtual environments, and the full dataset are intentionally excluded from Git tracking.

## Pipeline Description

### Step 1: Data Collection

Collects and organizes Gujarati handwritten text images and corresponding ground-truth transcriptions.

```bash
python step1_data_collection.py
```

### Step 2: Binarization

Converts grayscale handwritten images into binary images to improve text-background separation.

```bash
python step2_binarization.py
```

### Step 3: Noise Removal

Applies filtering to remove scanning noise, ink artifacts, and unwanted background texture.

```bash
python step3_noise_removal.py
```

### Step 4: Deskewing

Detects and corrects skew in handwritten text images to improve downstream segmentation and recognition.

```bash
python step4_deskew.py
```

### Step 5: Line Segmentation

Segments preprocessed handwritten pages into line-level image samples.

```bash
python step5_line_segmentation.py
```

### Step 6: ResNet-50 Feature Extraction

Extracts high-dimensional visual features from segmented Gujarati handwritten text images using a ResNet-50 backbone.

```bash
python step6_line_feature_extraction_split.py
```

### Step 7: Token and Manifest Preparation

Builds vocabulary files, target token sequences, and train/validation/test manifests.

```bash
python step7.py
```

### Step 8: Transformer Training

Trains the Transformer-based recognition model using extracted ResNet-50 features.

```bash
python step8.py
```

### Step 9: Beam Search Inference and Evaluation

Loads the trained checkpoint, performs beam-search decoding, and evaluates CER and WER.

```bash
python step9.py
```

### Translation GUI

Runs the Gujarati translation interface.

```bash
cd TRANSLATION_MODULE
python translation_gui.py
```

## Installation

Create and activate a Python environment:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA version from the official PyTorch installation instructions.

## Dataset

The full dataset is not committed directly to this repository because it is large and contains many image files. For this project, the local dataset folder is approximately 2.75 GB with about 190k files, which is not suitable for a normal GitHub repository commit.

The full dataset is available on Kaggle:

[Handwritten Gujarati Text Images](https://www.kaggle.com/datasets/mehulkumardalwadi/handwritten-gujarati-text-images)

Expected local layout after downloading the dataset:

```text
Dataset/
├── final_line_dataset.csv
├── file/
│   ├── train_images.txt
│   ├── train_labels.txt
│   ├── val_images.txt
│   ├── val_labels.txt
│   ├── test_images.txt
│   └── test_labels.txt
└── ...
```

## Results

The proposed system reports strong handwritten Gujarati recognition performance:

| Metric | Value |
| --- | ---: |
| Character Error Rate (CER) | 4.10% |
| Word Error Rate (WER) | 10.18% |
| Accuracy | 99.70% |
| Precision | 99.20% |
| Recall | 98.90% |
| F1-score | 99.05% |

The ablation study shows the benefit of combining ResNet-50 features, Transformer contextual modeling, and beam-search decoding:

| Model Setting | CER | WER |
| --- | ---: | ---: |
| ResNet-50 + Greedy Autoregressive Decoder | 0.115 | 0.265 |
| ResNet-50 + Transformer Decoder | 0.095 | 0.225 |
| ResNet-50 + Transformer Encoder + Greedy Decoding | 0.078 | 0.190 |
| ResNet-50 + Transformer Encoder + Beam Decoding | 0.041 | 0.1018 |

## Citation

If you use this work, please cite:

```bibtex
@misc{dalwadi2026gujarati_htr_translation,
  title  = {Bridging Regional and Global Languages: Handwritten Gujarati Text Recognition and Translation Using Deep Neural Network},
  author = {Dalwadi, Mehulkumar and Mehta, Abhishek},
  year   = {2026}
}
```

## License

Add a license before making the repository public. For academic code, MIT is commonly used for source code. Dataset licensing should be decided separately, especially if handwritten samples were collected from participants.
