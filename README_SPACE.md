---
title: Chest X-Ray Pneumonia Detection
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.8.0
app_file: app.py
pinned: false
license: mit
---

# 🏥 Chest X-Ray Pneumonia Detection

AI-powered tool for detecting pneumonia from chest X-ray images using deep learning.

## 🎯 Features

- **Real-time Analysis**: Upload X-ray images and get instant predictions
- **High Accuracy**: 97.71% test accuracy with 0.9983 ROC AUC
- **User-Friendly**: Clean, intuitive Gradio interface
- **Confidence Scores**: See probability for both Normal and Pneumonia classes
- **Educational Tool**: Perfect for learning about medical AI applications

## 📊 Model Performance

**Test Set Results:**
- Overall Accuracy: 97.71%
- ROC AUC Score: 0.9983
- Normal Detection Recall: 99.51%
- Pneumonia Detection Recall: 97.08%

**Per-Class Metrics:**

| Class      | Precision | Recall | F1-Score |
|------------|-----------|--------|----------|
| Normal     | 92.24%    | 99.51% | 95.73%   |
| Pneumonia  | 99.82%    | 97.08% | 98.43%   |

## 🛠️ Technical Details

**Architecture:** ResNet-50 (Transfer Learning)  
**Framework:** PyTorch  
**Dataset:** 5,232 chest X-ray images (70/15/15 train/val/test split)  
**Classes:** Normal, Pneumonia  
**Input Size:** 224x224 RGB images

**Training Details:**
- Optimizer: Adam (lr=0.0001)
- Loss: CrossEntropyLoss with class weights (1.94 for Normal, 0.67 for Pneumonia)
- Data Augmentation: Random flip, rotation, color jitter
- Early stopping with patience of 5 epochs
- Learning rate scheduling with ReduceLROnPlateau

## 🚀 How to Use

1. Upload a chest X-ray image (JPEG or PNG)
2. Click "Analyze X-Ray"
3. View prediction results and confidence scores
4. Read the detailed analysis and recommendations

**Try the example images** provided to see the model in action!

## ⚠️ Important Disclaimer

**THIS IS AN EDUCATIONAL TOOL ONLY - NOT FOR MEDICAL DIAGNOSIS**

- This application is for demonstration and educational purposes
- It is NOT a medical device and is NOT FDA approved
- Do NOT use for clinical decision-making
- Always consult qualified healthcare professionals for medical advice
- The creators assume no liability for misuse of this tool

## 🎓 Educational Value

This project demonstrates:
- End-to-end deep learning pipeline
- Transfer learning with pre-trained models
- Handling class imbalance in medical imaging
- Proper train/validation/test methodology
- Model evaluation with multiple metrics
- Production deployment of ML models

## 📚 About the Project

**Created by:** Emmanuel  
**Education:** MSc in Data Science  

This project showcases professional ML engineering practices:
- ✅ Proper data preprocessing and stratified splits
- ✅ Addressing class imbalance with weighted loss
- ✅ Comprehensive model evaluation
- ✅ Clean, documented code
- ✅ Production-ready deployment

## 🔗 Links

- **GitHub Repository:** [Add your repo link]
- **LinkedIn:** [Add your LinkedIn]
- **Portfolio:** [Add your portfolio]

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) by Paul Mooney
- Architecture: ResNet-50 (He et al., 2015)
- Framework: PyTorch team

---

**Note:** If you use this project or find it helpful, please consider giving it a ⭐ star on GitHub!
