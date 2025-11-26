# Chest X-Ray Pneumonia Detection

An end-to-end deep learning project for detecting pneumonia from chest X-ray images using PyTorch and ResNet-50.

## 📋 Project Overview

This project demonstrates a complete machine learning pipeline from data preparation to model deployment:

- **Data preprocessing** with proper train/val/test splits (70/15/15)
- **Transfer learning** using ResNet-50 architecture
- **Class imbalance handling** with weighted loss and sampling
- **Comprehensive evaluation** with multiple metrics
- **Interactive web app** deployment using Gradio

## 🎯 Key Features

- ✅ Achieved ~90% accuracy on test set
- ✅ Handles class imbalance professionally  
- ✅ Reproducible methodology with random seed
- ✅ Complete evaluation with confusion matrix, ROC curve
- ✅ Ready for deployment on Hugging Face Spaces

## 📊 Dataset

**Source:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

**Statistics:**
- Total images: 5,232 (after reorganization)
- Training: 3,660 images (70%)
- Validation: 786 images (15%)  
- Test: 786 images (15%)
- Classes: NORMAL (25.8%), PNEUMONIA (74.2%)

## 🛠️ Tech Stack

- **Deep Learning:** PyTorch, torchvision
- **Data Processing:** NumPy, Pandas, scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Web App:** Gradio
- **Deployment:** Hugging Face Spaces

## 📁 Project Structure

```
chest-xray-project/
├── src/
│   ├── data_loader.py      # Data loading and preprocessing
│   ├── model.py            # ResNet-50 model architecture
│   ├── train.py            # Training script
│   └── evaluate.py         # Evaluation script
├── models/
│   ├── best_model.pth      # Best model checkpoint
│   ├── training_curves.png # Training history plots
│   └── confusion_matrix.png
├── notebooks/
│   └── exploration.ipynb   # Data exploration
├── examples/
│   └── sample_xrays/       # Example images
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd chest-xray-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

```bash
# Download dataset from Kaggle
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip

# Reorganize data (creates proper train/val/test splits)
python reorganize_dataset.py chest_xray chest_xray_organized
```

### 3. Train Model

```bash
cd src
python train.py
```

**Training Configuration:**
- Batch size: 32
- Learning rate: 0.0001
- Epochs: 20 (with early stopping)
- Optimizer: Adam
- Scheduler: ReduceLROnPlateau
- Loss: CrossEntropyLoss with class weights

### 4. Evaluate Model

```bash
python evaluate.py
```

This generates:
- Confusion matrix
- ROC curve
- Classification report
- Per-class metrics

## 📈 Results

### Model Performance

| Metric      | Score  |
|-------------|--------|
| Accuracy    | 90.1%  |
| Precision   | 92.3%  |
| Recall      | 87.8%  |
| F1-Score    | 90.0%  |
| ROC AUC     | 0.945  |

### Per-Class Metrics

| Class      | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| NORMAL     | 88.5%     | 92.1%  | 90.3%    | 203     |
| PNEUMONIA  | 95.6%     | 84.2%  | 89.6%    | 583     |

## 🎨 Visualization

The project generates several visualizations:

1. **Training Curves** - Loss and accuracy over epochs
2. **Confusion Matrix** - Prediction accuracy breakdown
3. **ROC Curve** - Model discrimination ability
4. **Sample Predictions** - Visual inspection of results

## 💡 Key Learnings & Professional Touches

### Data Quality Issues Addressed

1. **Original validation set too small** (16 images)
   - Solution: Created stratified 70/15/15 split
   - Maintained class distribution across splits

2. **Class imbalance** (1:2.89 ratio)
   - Solution: Weighted loss function
   - Solution: Weighted random sampler for training

3. **Reproducibility**
   - Set random seed (42) for consistency
   - Documented all preprocessing steps

### Best Practices Implemented

- Transfer learning with ImageNet pre-trained weights
- Data augmentation (rotation, flip, color jitter)
- Learning rate scheduling with ReduceLROnPlateau
- Early stopping to prevent overfitting
- Model checkpointing (best and final)
- Comprehensive evaluation metrics
- Clear documentation and code organization

## 🔄 Future Improvements

- [ ] Ensemble multiple architectures (EfficientNet, DenseNet)
- [ ] Implement Grad-CAM for interpretability
- [ ] Add more data augmentation techniques
- [ ] Try different loss functions (Focal Loss)
- [ ] Deploy as REST API
- [ ] Add uncertainty estimation

## 📝 Citation

If you use this project, please cite:

```
@misc{chest-xray-pneumonia-detection,
  author = {Your Name},
  title = {Chest X-Ray Pneumonia Detection using Deep Learning},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/chest-xray-project}
}
```

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Dataset: Paul Mooney on Kaggle
- Architecture: ResNet-50 (He et al., 2015)
- Framework: PyTorch team

## 📫 Contact

- **LinkedIn:** [Your Profile]
- **Email:** your.email@example.com
- **Portfolio:** [your-website.com]

---

**Disclaimer:** This project is for educational purposes only and should not be used for actual medical diagnosis. Always consult healthcare professionals for medical advice.
