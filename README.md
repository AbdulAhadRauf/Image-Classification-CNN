# Deep Learning Practice Week 9 – Image Classification of Flora & Fauna

## 📄 Competition Link

[Deep Learning Practice Week 9 on Kaggle](https://www.kaggle.com/competitions/deep-learning-practice-week-9-image-c-lassifica/data)

## 🏆 Overview

As part of the Deep Learning Practice (DLP) course, build a convolutional neural network that classifies images into 10 biological categories (animals, plants, fungi, etc.) with the highest possible weighted F1 score.

* **Task:** Image classification into 10 classes
* **Start:** March 7, 2025
* **Close:** March 19, 2025 (05:00 PM IST)
* **Evaluation:** Weighted F1 score
* **Submission:** Kaggle CSV + Jupyter notebook via Google Form

## 🔖 Class Mapping

| Class Name | Label ID |
| ---------- | -------: |
| Amphibia   |        0 |
| Animalia   |        1 |
| Arachnida  |        2 |
| Aves       |        3 |
| Fungi      |        4 |
| Insecta    |        5 |
| Mammalia   |        6 |
| Mollusca   |        7 |
| Plantae    |        8 |
| Reptilia   |        9 |

## 🗃️ Dataset Structure

```
├── data/
│   ├── train/
│   │   ├── Amphibia/    # 1,000 images
│   │   ├── Animalia/    # 1,000 images
│   │   └── ...          # total 10 folders
│   └── test/            # 2,000 unlabeled images
├── notebooks/           # EDA & training notebooks
├── src/                 # data loaders, model definitions, utils
├── outputs/             # checkpoints, logs, plots
├── submission.py        # script to generate ROLL_NO.csv
└── README_DLP_Image_Classification.md  # This file
```

## 🛠️ Environment & Dependencies

* **Python 3.8+**
* **PyTorch** & **Torchvision**
* **TensorFlow/Keras** (optional)
* **scikit-learn** (metrics, train/test split)
* **NumPy**, **Pandas**
* **OpenCV**, **Pillow**
* **albumentations** (image augmentations)
* **Matplotlib**, **Seaborn**
* **tqdm**

Install via:

```bash
pip install torch torchvision numpy pandas opencv-python pillow albumentations scikit-learn matplotlib seaborn tqdm
```

## 🔍 Exploratory Data Analysis (EDA)

1. **Class Balance:** bar chart of image counts per class
2. **Sample Grid:** visualize representative images from each category
3. **Image Statistics:** distribution of resolutions, mean RGB values
4. **Augmentation Preview:** examples of flips, rotations, color jitter

## 🏗️ Model Architectures & Approach

### 1. Simple CNN Baseline

* **Architecture:** 4–6 convolutional layers + pooling + batch normalization + dropout
* **Head:** fully connected layers ending in softmax for 10 classes
* **Loss:** `CrossEntropyLoss()`
* **Optimizer:** `Adam(lr=1e-3)`

### 2. Transfer Learning with ResNet50

* **Backbone:** pretrained ResNet50 (ImageNet)
* **Modification:** replace final FC with 10-unit classification head
* **Fine-tuning:** freeze backbone for initial epochs, then unfreeze last block
* **Loss & Optimizer:** as above, with smaller lr for pretrained weights (1e-4)

### 3. EfficientNet-B0

* **Backbone:** EfficientNet-B0 (pretrained)
* **Head:** same replacement strategy as ResNet50
* **Advantages:** balanced performance & efficiency

#### Common Techniques

* **Data Augmentation:** random crop, horizontal/vertical flips, color jitter
* **Learning Rate Scheduler:** `StepLR` or `CosineAnnealingLR`
* **Regularization:** dropout, weight decay
* **Mixed Precision:** optional `torch.cuda.amp` for faster training

## 🎓 Training & Validation Pipeline

1. **DataLoader:** stratified split (80% train, 20% val)
2. **Batch Size:** 32 (adjust based on GPU memory)
3. **Epochs:** 30–50 with early stopping on validation F1 score
4. **Metrics:** compute weighted F1 per epoch using `sklearn.metrics.f1_score`
5. **Checkpointing:** save best model by validation weighted F1

## 📊 Evaluation & Results

* **Primary Metric:** weighted F1 score on validation/test set
* **Confusion Matrix:** inspection of per-class errors
* **Learning Curves:** training vs. validation loss & F1

## 🚀 Inference & Submission

1. **Prepare Test Predictions:** load best checkpoint, run inference on `data/test`
2. **Generate CSV:** `Image_ID,Label` rows, save as `ROLL_NO.csv` (uppercase roll number)
3. **Submit:** upload CSV to Kaggle; submit notebook & this README via Google Form

```bash
python submission.py --model resnet50 --weights best_resnet50.pth \
    --test_dir data/test --output ROLL_NO.csv
```

## 🏃 How to Reproduce

1. Clone this repo and set your roll number in `submission.py`.
2. Install dependencies.
3. Place data under `data/`.
4. Run EDA notebook: `notebooks/eda.ipynb`.
5. Train models in `notebooks/train_*.ipynb`.
6. Execute `submission.py` to create submission file.

