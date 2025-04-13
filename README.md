# My Project

# Facial Expression Recognition with CNN

This project is a Convolutional Neural Network (CNN) based facial expression recognition model built using **Keras** and trained on grayscale images of size `48x48`. It classifies faces into **7 emotion categories** such as happy, sad, angry, etc.

---

## 🧠 Model Architecture

The CNN architecture consists of:

- Multiple **Conv2D + MaxPooling + Dropout** layers for feature extraction.
- **Flatten + Dense** layers for classification.
- Output layer with `softmax` activation for multiclass prediction.

---

## 📁 Dataset Structure

Your dataset should be organized as:

images/ ├── train/ │ ├── angry/ │ ├── happy/ │ ├── disgust/ │ ├── surprise/ │ ├── fear/ │ ├── neutral/ │ └── sad/ └── test/ ├── angry/ ├── happy/ ├── disgust/ ├── surprise/ ├── fear/ ├── neutral/ └── sad/


- Images are grayscale with size `48x48`.
- Folder names represent the emotion labels.

---

## 🔧 Installation

Make sure you have the following libraries installed:

```bash
pip install tensorflow keras numpy pandas pillow scikit-learn tqdm
```
## 🚀 How to Run
- Clone the Repo:
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name

```
- Add your dataset inside the images/ directory as shown above
- Run the Python script

```bash
python your_script_name.py
```

## 🧪 Training
- Normalizes pixel values by dividing by 255.

- Labels are encoded and one-hot encoded using LabelEncoder and to_categorical.

- Model is trained using:

    - Adam optimizer

    - Categorical Crossentropy loss

    - Accuracy metric

    - 100 epochs

    - Batch size of 128

    - 20% validation split

## 📊 Performance
- The model achieves decent performance on facial emotion recognition with further improvements possible by:

    - Data augmentation

    - More training data

    - Hyperparameter tuning

    - Transfer learning with pretrained models

## 📌 Key Dependencies
- TensorFlow / Keras
- NumPy
- Pandas
- scikit-learn
- PIL (Pillow)
- tqdm

## 💬 Acknowledgments
- Inspired by facial emotion recognition challenges like FER2013 and real-world applications in emotion AI and HCI.