# Classroom Object Classification

**Group Number:** 2  
**Group Members:** Salman Alharbi, Hussain Aljubran, Ali Alkhathami, Abdullah Aljammaz  

---

## Introduction

**Problem:**  
Automatically classifying classroom objects is challenging due to inconsistent image quality, lighting, and overlapping items.  

**Dataset:**  
~3,400 images (~425 per class) across 8 categories, collected from different angles and lighting, preprocessed to 240×240 resolution.  

**Goal:**  
Achieve high accuracy in classifying classroom objects, demonstrating the feasibility of computer vision for classroom automation.  

---

## Data & Preprocessing

**Dataset:**  
- 8 object classes: PC, Monitor, Mouse, Keyboard, Whiteboard, Chair, Mobile, Bag  
- ~3,400 images (~425 per class)  
- **Download dataset:** [![Dataset](https://img.shields.io/badge/Download-Dataset-blue?style=for-the-badge)](https://drive.google.com/drive/folders/1E8K2VCqY2ZJvqAI_A4_FkacGPF2CmR3U?usp=share_link)  

**Collection:**  
- Multiple angles, lighting, and distances to increase variety  

**Cleaning:**  
- Removed non-image files and ensured valid formats (.jpg)  

**Normalization:**  
- Resized all images to 240×240 pixels for consistency  

**Labeling:**  
- Renamed files systematically (e.g., `mouse_001.jpg`)  

**Deployment (Streamlit):**  
- Framework: Streamlit for interactive web deployment  
- Interface: Upload an image → get predicted class  
- Accessibility: QR Code for instant access  
- Design: Custom background and simple UI  

**Note:**  
For more details on the dataset, see [DATASET.md](DATASET.md)

---

## Methods

**Models:**  
- 3 Custom CNN models and Transfer Learning with pretrained models  

**Tuning:**  
- Optimizer: Adam with learning rate = 1e-5  
- Early stopping (patience = 3)  
- Checkpointing at each epoch  
- Evaluation Metric: Accuracy on validation set  

**Note:**  
For more details on the pretrained models, see [Module.md](Module.md)

---

## Results

- Preprocessing produced a balanced and consistent dataset of ~3,400 images  
- Dataset analysis (graphs for brightness, contours, distribution) confirmed quality and readiness  
- **Validation Accuracy (VGG16 Model):** ~99%  

---

## Discussion

- A strong dataset improved the chances of better model accuracy  
- Planned advanced augmentation (rotations, flips, color shifts) but could not implement due to time limits  
- The model required strong hardware (GPU/CPU and memory) to process efficiently  
- Fine-tuning the last 4 convolutional layers improved performance over using frozen VGG16  

---

## Conclusion & Future Work

**Conclusion:**  
We successfully built a CNN-based system that classifies eight classroom objects with good performance, demonstrating the potential of computer vision in real-world environments.  

**Future Work:**  
Our future goal is to transform the current app into an interactive educational tool for young children, including those with learning difficulties, where the model can recognize classroom objects, pronounce their names, and provide interactive exercises, while gradually collecting real-world data to expand the dataset and improve classification performance for broader educational use.
