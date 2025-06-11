## Face Mask Detection using Faster R-CNN

This project implements a deep learning-based *face mask detection system* using *Faster R-CNN with a ResNet-50 FPN backbone*. The model identifies whether people in images are wearing masks correctly, incorrectly, or not at all — a crucial task in public health monitoring and safety enforcement during pandemics.

---

##  Project Highlights

-  Built using *PyTorch* and *Torchvision’s object detection API*
-  Utilized a publicly available dataset with bounding box annotations
-  Detected 3 categories: with_mask, without_mask, and mask_weared_incorrect
- Applied image augmentation using *Albumentations*
-  Visualized predictions with bounding boxes on test images
-  Fine-tuned a pre-trained *Faster R-CNN ResNet50-FPN*

---

##  Dataset

- *Name*: Face Mask Detection Dataset
- *Source*: [Kaggle - Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
- *Structure*:
  - /images: RGB face images
  - /annotations: Pascal VOC-style .xml annotation files
- *Classes*:
  - with_mask
  - without_mask
  - mask_weared_incorrect

---

##  Model Architecture

- Base model: Faster R-CNN with ResNet50 + FPN from torchvision.models.detection
- Modified the classifier head to predict 3 categories
- Outputs:
  - Bounding boxes
  - Class labels
  - Confidence scores

---

##  Preprocessing & Augmentation

- Parsed .xml annotations using Python’s ElementTree
- Created a custom PyTorch Dataset class for loading and formatting data
- Augmentation pipeline with Albumentations:
  - HorizontalFlip
  - Resize
  - Normalize
  - Convert to tensor (ToTensorV2)

---

##  Model Training

- *Optimizer*: SGD (Stochastic Gradient Descent)
- *Scheduler*: StepLR
- *Loss Function*: Combined classification + bounding box regression loss
- *Epochs*: 10 (adjustable)
- *Batch Size*: 4
- *Device*: GPU (recommended)

---

##  Results

- Achieved visually accurate detection of all 3 face mask classes.
- Bounding boxes and predictions were successfully plotted using Matplotlib.
- Real-world examples tested with good generalization.


---

##  Output Visualization

The model outputs include:
- Class label (with_mask, without_mask, mask_weared_incorrect).
- Bounding box around detected face.
- Confidence score for each prediction.

Example visualization:

<img src="face mask detection output.png" alt="Face mask detection" width="700"/>

---

## How to Run

###  Clone the Repository

```bash
git clone https://github.com/yourusername/face-mask-detection.git
cd face-mask-detection
pip install -r requirements.txt
jupyter notebook Face_Mask_Detection.ipynb
```


## Dependencies
Python 3.x

PyTorch

Torchvision

Albumentations

Matplotlib, Seaborn

PIL, OpenCV

tqdm, ElementTree

Google Colab (for GPU acceleration)

---

## Author
Pakhi Singhal
