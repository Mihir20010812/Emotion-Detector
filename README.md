# Emotion-Detector

This project is a **Facial Emotion Detection System** built using deep learning. It recognizes human emotions from facial expressions in real-time or from images using a convolutional neural network (CNN).

## Overview

The model classifies emotions into seven categories:

* Angry
* Disgust
* Fear
* Happy
* Sad
* Surprise
* Neutral

The system uses a **Convolutional Neural Network (CNN)** trained on grayscale facial images (48x48 pixels). The trained model is stored as:

* `facialemotionmodel.json` – Model architecture
* `facialemotionmodel.h5` – Model weights

The training process and evaluation are included in `trainmodel.ipynb`.


##  Model Architecture

The model consists of:

* Multiple **Conv2D** layers with ReLU activation
* **MaxPooling2D** and **Dropout** layers to prevent overfitting
* **Dense** fully-connected layers with softmax output

Frameworks used:

* **TensorFlow / Keras**
* **OpenCV** (for face detection)
* **NumPy**
* **Matplotlib** (for visualization)

##  Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/emotion-detector.git
   cd emotion-detector
   ```

2. Install dependencies:

   ```bash
   pip install tensorflow opencv-python numpy matplotlib
   ```

3. (Optional) Launch Jupyter Notebook:

   ```bash
   jupyter notebook trainmodel.ipynb
   ```


##  Usage

### Run on an image

```python
python detect_emotion.py --image path_to_image.jpg
```

### Run with webcam

```python
python detect_emotion.py --webcam
```

The script will open a window showing the detected face(s) and their predicted emotion labels.



##  Results

The model achieves around **X% accuracy** on the validation dataset (replace with your metric).
Example output:

| Input Image                     | Predicted Emotion |
| ------------------------------- | ----------------- |
| ![face](assets/sample_face.jpg) | Happy             |



##  Project Structure


emotion-detector/
│
├── facialemotionmodel.json       # Model architecture
├── facialemotionmodel.h5         # Trained weights
├── trainmodel.ipynb              # Training notebook
├── detect_emotion.py             # (Optional) real-time detector script
├── README.md                     # Project documentation
└── requirements.txt              # Dependencies
```

