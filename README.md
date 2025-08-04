# DermaVision

DermaVision is a machine learning-powered diagnostic tool that addresses the critical shortage of dermatological care in Canada. The system combines advanced image analysis with patient metadata to provide accessible preliminary skin condition assessments, helping to reduce wait times and improve healthcare access in underserved areas.

[Project Report](https://github.com/nina2dv/DermaVision/blob/main/Project%20Report_%20DermaVision%20(Team%20Salus).pdf)
[Slides](https://github.com/nina2dv/DermaVision/blob/main/DermaVision_%20Classifying%20and%20Diagnosing%20Skin%20Cancer.pdf)

## Problem Statement
- Skin diseases represent the 4th leading cause of non-fatal disease burden worldwide
- Canada has fewer than 700 licensed dermatologists serving ~38 million people
- Wait times extend beyond a year in major metropolitan areas
- Rural communities face disproportionate access barriers

## Solution Overview
DermaVision uses a mixed-input neural network model that processes both dermoscopic images and clinical metadata to classify skin lesions as benign or malignant. The system uses:
- CNN Branch: MobileNetV2-based image processing for dermoscopic images
- MLP Branch: Clinical metadata analysis
- Image Segmentation: Otsu thresholding and Gabor filters

## Dataset
- International Skin Imaging Collaboration (ISIC) Dataset
- Original size: 503,955 images with clinical metadata
- Features: 30 metadata fields → 5 essential features selected via Decision Tree analysis

## Architecture
Mixed-Input Model:
```
Input Images (224×224×3) ---> CNN Branch --|
                                           ├-> Concatenation --> Final Layers --> Classification
Clinical Metadata ----------> MLP Branch --|
```

### CNN Branch:
- MobileNetV2 (frozen, pre-trained on ImageNet)
- Global Average Pooling
- Dense layer (256 neurons) + Dropout (0.5)

### MLP Branch:
- Dense layer (128 neurons) + BatchNorm + Dropout (0.3)
- Dense layer (64 neurons) + BatchNorm + Dropout (0.3)

### Final Layers:
- Concatenated features (320 dimensions)
- Dense layer (128 neurons) + BatchNorm + Dropout (0.5)
- Output layer (1 neuron, sigmoid activation)


## Dependencies
```bash
pip install tensorflow>=2.8.0
pip install opencv-python
pip install scikit-learn
pip install pandas numpy
pip install matplotlib seaborn
pip install jupyter
```

## Results
- Best Overall Performance: Non-segmentation model (85% accuracy)
- Class Imbalance Impact: Consistent performance gap between benign and malignant classification
- Overfitting Indicators: Gap between training and validation metrics observed
- Skin Type Bias: Limited representation of darker skin tones in dataset (missing Fitzpatrick skin types V-VI)
- Feature Correlation: Weak correlations between metadata features

## Future Enhancements
- Model Refinement: Address overfitting and improve generalization
- Explainability/Transparency: Implement model interpretability features
- Subtyping: Identify specific types of melanoma and benign conditions
- Diverse Dataset: Include more representative skin tone samples
- Clinical Integration: Develop user-friendly interface for healthcare practitioners
