# **Xray Xplorer** · CNN, XGBoost & Grad-CAM · COVID-19 Detection in Chest X-Ray Images Using Explainable Boosting Algorithms

Welcome to the GitHub repository for "Xray Xplorer", a powerful diagnostic tool birthed from a dissertation project titled "COVID-19 Detection in Chest X-Ray Images using Explainable Boosting Algorithms". The dissertation addresses the urgent demand for transparency in AI-powered diagnostic models. Uniting the robustness of Convolutional Neural Networks (CNNs), the potency of eXtreme Gradient Boosting (XGBoost), and the transparency of Gradient-weighted Class Activation Mapping (Grad-CAM), Xray Xplorer provides precise and interpretable COVID-19 predictions using chest X-ray images. Proven to perform admirably with an accuracy of 94.05% and an F1 score of 94.08%, this tool is ready to contribute to the fight against the pandemic.

<p align='center'><img src='https://github.com/Abhijeet-Pitumbur/Xray-Xplorer/blob/main/screenshots/screenshot-1.png'/></p>
<p align='center'><img src='https://github.com/Abhijeet-Pitumbur/Xray-Xplorer/blob/main/screenshots/screenshot-2.png'/></p>
<p align='center'><img src='https://github.com/Abhijeet-Pitumbur/Xray-Xplorer/blob/main/screenshots/screenshot-3.png'/></p>

Xray Xplorer presents a robust interface for healthcare professionals to upload chest X-ray images, returning a diagnostic prediction of whether the image indicates a normal condition, pneumonia (viral or bacterial), or COVID-19. The web application offers a Grad-CAM visualization alongside the prediction, fostering explainability and interpretability - key requirements for trust in AI-driven diagnostic tools.

### [View Xray Xplorer](https://xrayxplorer.com)

##### [Download Repository](https://github.com/Abhijeet-Pitumbur/Xray-Xplorer/archive/refs/heads/main.zip)  · GitHub

This repository contains the source code for the Flask-based web application, designed to facilitate the deployment and use of the diagnostic AI model trained in the project's Jupyter notebook on Google Colab.

##### [View Jupyter Notebook](https://colab.research.google.com/github/Abhijeet-Pitumbur/Xray-Xplorer/blob/main/Xray-Xplorer.ipynb)  · Google Colab

The complete dissertation report, outlining the comprehensive research, methodology, and broader implications, is available for deeper insights into the project.

##### [View Dissertation](https://bit.ly/abhijt-xray-xplorer-report)  · Google Drive

All model files, including individual CNN models and the final hybrid CNN-XGBoost model, are available in the Google Drive linked below.

##### [View Model Files](https://bit.ly/abhijt-xray-xplorer-models)  · Google Drive

## Prerequisites and Installation

To get this project running in your local environment, you will need Python 3 installed.

##### [Download Python 3](https://www.python.org/downloads/release/python-31011)  · Python.org

Then, clone or download the repository, open the *project* folder in your terminal, and follow these steps:

- Create a Python virtual environment, for isolation:

```
python3 -m venv venv
```

- Activate the virtual environment:

```
# Windows:
venv\Scripts\activate

# Linux and macOS:
source venv/bin/activate
```

- Install the necessary Python packages:

```
pip install -r requirements.txt
```

- Set the Flask app environment variable:

```
# Windows:
set FLASK_APP=app.py

# Linux and macOS:
export FLASK_APP=app.py
```

- Run the application:

```
flask run
```

The application should now be running at [localhost:5000](http://localhost:5000)

## Model Overview

Dive into the world of hybrid modeling that combines the image processing finesse of CNNs, the robust ensemble learning mechanism of XGBoost, and the interpretability aspect of Grad-CAM. This unique integration forms a comprehensive, efficient, and transparent tool for COVID-19 detection.

**Convolutional Neural Networks (CNNs):** CNNs excel in processing grid-like data, such as image data. A CNN uses filters and different layers to automatically extract and learn features from an image which is fed into the network in the raw form.

**XGBoost:** XGBoost is an optimized distributed gradient boosting library, designed to be highly efficient, flexible, and portable. Gradient boosting is a machine learning technique where the model learns from its mistakes in each iteration. XGBoost enhances the model's performance by merging the outputs of many weak learners.

**Grad-CAM:** Grad-CAM, short for Gradient-weighted Class Activation Mapping, provides visual explanations for model predictions. It generates a heatmap highlighting the important regions in the input image that led to the model's decision.

Embark on the journey through the methodical steps involved in creating the model:

1. **Data Preprocessing**: Normalizes and standardizes input images to enhance model learning.
2. **Data Augmentation**: Enlarges the training set through techniques like rotation, flipping, brightness variation, and zooming to boost model's generalization ability.
3. **Dataset Preparation**: Uses ImageDataGenerator instances to divide and stream data during model training, validation, and testing.
4. **CNN Model Implementation**: Adapts pre-trained CNN models (VGG16, ResNet50, InceptionV3) for feature extraction and serves these features as input to the XGBoost model for classification.
5. **XGBoost Hyperparameter Tuning and Feature Selection**: Applies Bayesian optimization with cross-validation for optimal hyperparameter selection, concurrently performing feature selection to hone in on the most predictive features.
6. **CNN-XGBoost Hybrid Model Evaluation and Testing**: Evaluates the performance of the hybrid model using metrics like accuracy, precision, recall, F1 score, specificity, and AUC-ROC.
7. **Grad-CAM Implementation**: Offers insights into the decision-making process of the CNN model through heatmap visualizations.
8. **Chest X-Ray Image Verification**: Uses a specially-trained VGG16 model to ensure that the uploaded image is a chest X-ray before proceeding with the COVID-19 detection pipeline, thereby mitigating errors from incorrect image uploads.

Together, these methods construct a model that is not only efficient and accurate but also interpretable, lending insight into its diagnostic process.

## Dataset Overview

This project utilizes the "COVID-QU-Ex Dataset" (Coronavirus Disease Qatar University Extended Dataset) from Kaggle, encompassing 33,900 chest X-ray images, categorized as COVID-19 positive cases, normal cases, and non-COVID infection cases.

The distribution of the images in the dataset is as follows:

- 11,950 images are classified as COVID-19 positive cases,
- 10,695 images are classified as normal cases,
- 11,255 images are classified as positive cases for non-COVID infections.

The dataset is divided into training, validation, and testing sets containing 21,705 images, 5,410 images, and 6,785 images respectively.

#### [View Dataset](https://doi.org/10.34740/KAGGLE/DSV/3122958)  · Kaggle

## Languages, Frameworks and Tools
- Python 3.10.11
- TensorFlow 2.12.0
- Keras 2.12.0
- XGBoost 1.7.6
- OpenCV 4.7.0
- Flask 2.3.2
- jQuery 3.7.0
- Visual Studio Code 1.80
- Google Colab

## Credits
- Abhijeet Pitumbur