# Face Recognition Using Traditional Computer Vision Techniques

## 📌 Project Overview

This project implements a face recognition system using traditional computer vision and machine learning techniques. It combines three feature extraction methods — **Local Binary Patterns (LBP), Histogram of Oriented Gradients (HOG), and Gabor Filters** — to capture facial texture, edge, and spatial information.

The extracted features are standardized, reduced using Principal Component Analysis (PCA), and classified using a Support Vector Machine (SVM). Hyperparameter tuning is performed using GridSearchCV to identify a suitable model configuration.

The project uses image preprocessing and data augmentation to improve the diversity of training samples.

## 🎯 Objectives

- Implement face recognition using traditional computer vision methods.
- Extract complementary facial features using LBP, HOG, and Gabor Filters.
- Improve training data diversity through image augmentation.
- Reduce feature dimensionality using PCA.
- Train and optimize an SVM classifier.
- Generate predictions for test images and export them to a CSV file.

## 🛠️ Technologies Used

- **Python**
- **OpenCV** — Image processing and preprocessing
- **NumPy** — Numerical operations
- **Pandas** — Dataset and CSV handling
- **Scikit-image** — HOG and LBP feature extraction
- **Scikit-learn** — Label encoding, feature scaling, PCA, SVM, and hyperparameter tuning
- **Matplotlib** — Visualization support
- **Google Colab** — Notebook execution and Google Drive integration

## ⚙️ Methodology

### 1. Data Loading and Label Encoding

The training labels are loaded from a CSV file. `LabelEncoder` converts the class labels into numerical values for model training.

### 2. Image Preprocessing

Each image is:
- Converted to grayscale.
- Resized to 128 × 128 pixels.
- Processed using histogram equalization to improve contrast.
- Normalized to the range [0, 1].

### 3. Data Augmentation

Training images are augmented using:
- Horizontal flipping
- Rotation by -10° and +10°
- Brightness adjustments
- Contrast enhancement using CLAHE
- Gaussian noise addition

These transformations create additional training samples and expose the model to variations in image appearance.

### 4. Feature Extraction

The project combines three traditional feature extraction techniques:

| Technique | Purpose |
|---|---|
| **HOG** | Captures facial edge directions and shape information. |
| **LBP** | Captures local texture patterns in facial regions. |
| **Gabor Filters** | Captures texture and directional responses at different orientations. |

The features from all three methods are concatenated into a single feature vector for each image.

### 5. Feature Scaling and Dimensionality Reduction

- `StandardScaler` standardizes the extracted features.
- `PCA` reduces the number of features while retaining 98% of the variance.

This reduces the dimensionality of the feature space before classification.

### 6. Model Training and Hyperparameter Tuning

A Support Vector Machine (SVM) classifier with an RBF kernel is trained using the extracted features.

`GridSearchCV` with three-fold cross-validation evaluates different combinations of:
- `C`: [1, 5, 10]
- `gamma`: [`scale`, 0.001, 0.01, 0.1]
- `kernel`: [`rbf`]

The best-performing configuration is selected based on cross-validation accuracy.

### 7. Prediction and Submission

The test images undergo the same preprocessing and feature extraction pipeline. The trained scaler and PCA transformation are applied before the SVM predicts the class labels.

The predicted numerical labels are converted back to their original class labels, and the results are exported to `submission.csv`.

## 📂 Dataset Structure

The notebook expects the dataset to be organized as follows:

```text
ECE501_Mugshots/
├── train/
│   ├── image1.jpg
│   └── ...
├── test/
│   ├── image2.jpg
│   └── ...
├── train_labels.csv
└── sample_submission.csv
```

- `train/`: Training images.
- `test/`: Test images requiring predictions.
- `train_labels.csv`: Training image filenames and corresponding class labels.
- `sample_submission.csv`: Test filenames used to generate the prediction file.

**Note:** The dataset is not included in this repository. The notebook currently uses Google Drive paths, which must be updated to match your dataset location.

## 🚀 How to Run the Project

### Option 1: Google Colab

1. Upload `Face_Recognition.ipynb` to Google Colab.
2. Place the dataset in Google Drive using the folder structure shown above.
3. Update the dataset paths in the notebook if necessary.
4. Run the notebook cells in sequence.
5. Check the generated `submission.csv` file.

### Option 2: Local Environment

1. Clone or download this repository.
2. Install the required Python libraries:

```bash
pip install opencv-python numpy pandas scikit-image scikit-learn matplotlib jupyter
```

3. Update the dataset paths to point to your local folders.
4. Remove or replace the Google Colab-specific Google Drive mounting code.
5. Run the notebook:

```bash
jupyter notebook Face_Recognition.ipynb
```

## 📊 Output

The project generates a `submission.csv` file containing the predicted labels for the test images.

```text
filename,label
image2.jpg,class_A
image3.jpg,class_B
```

The class names above are illustrative examples. Actual labels depend on the dataset and model predictions.

## 🔒 Data Privacy

The dataset is not included in the project repository. Ensure that facial images and associated labels are handled responsibly, with appropriate authorization and privacy safeguards.

## 📚 References

1. [IEEE Xplore Research Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4803030)
2. [Journal of College of Education for Pure Sciences Research Paper](https://jceps.utq.edu.iq/index.php/main/article/view/34/22)
3. [Google Colab Notebook — Individual Method Experiments](https://colab.research.google.com/drive/1TwPOm3aCgkt6S0dmeGscjRJcuWRQKWm3?usp=sharing)

## 👩‍💻 Project Information

**Project:** Face Recognition Using Traditional Computer Vision Techniques

**Approach:** Feature Fusion (LBP + HOG + Gabor), PCA, and SVM Classification

**Implementation:** Python and Jupyter Notebook
