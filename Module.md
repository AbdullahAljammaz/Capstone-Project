# Pretrained Models

This project uses several pretrained models for classroom object classification.  
These models were fine-tuned to classify 8 classroom objects: PC, Monitor, Mouse, Keyboard, Whiteboard, Chair, Mobile, and Bag.

## Models and Download Links

- **capResNet50:** [Download Link](https://drive.google.com/file/d/1zvtYbo92sjAgtgNzfbWaTiyAwM-VWBEW/view?usp=share_link)  
- **capResNet101:** [Download Link](https://drive.google.com/file/d/1inqcxDkiiBBQAp_XIQqHNd_gb-3fmHT-/view?usp=share_link)  
- **capVGG16:** [Download Link](https://drive.google.com/file/d/1KIj20nQ68_chvNFIm2PWXzYh2GIIAJRe/view?usp=share_link)  
- **capVGG19:** [Download Link](https://drive.google.com/file/d/1ygMe8Iv92r338vImfSB2kiWgpvJyInqS/view?usp=share_link)  

## Notes

- Fine-tuned the last 4 convolutional layers; earlier layers were frozen to preserve pretrained features.
- These models help improve classification performance and reduce training time compared to training from scratch.
- Any challenges encountered using pretrained models were mainly related to computational resources (GPU/CPU and memory usage).
