## Experimenting faster with No Code/Low Code ML tools from AWS

The Machine Learning (ML) journey requires continuous experimentation and rapid prototyping to be successful. In order to create highly accurate and performant models, data scientists have to first experiment with feature engineering, model selection and optimization techniques. These processes are traditionally time consuming and expensive. In this tech talk, data scientists will learn how the low-code / no-code capabilities found in **Amazon SageMaker Data Wrangler**, **Autopilot** and **Jumpstart**, make it easier to experiment faster and bring highly accurate models to production more quickly and efficiently.

* **What you will learn:**
    * Learn how to simplify the process of data preparation and feature engineering, and complete each step of the data preparation workflow.
    * Understand how to automatically build, train, and tune the best machine learning models based on your data, while allowing you to maintain full control and visibility.
    * Get started with ML easily and quickly using pre-built solutions for common use cases and open source models from popular model zoos.

## Representative Use Cases:
**Note:** If you are new to SageMaker Studio, instructions to get started with SageMaker Studio and setup this demo repository are provided [here](https://github.com/arunprsh/no-code-low-code/blob/main/setup-studio.md).

###  Demo I & II - Tabular (SageMaker DataWrangler + Autopilot)

* **Use case:** **Predict the onset of diabetes based on diagnostic measures**
    * This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.
* **Dataset:** 
    * The datasets consists of several medical predictor variables and one target variable, Outcome. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.
    * x rows, y feature columns and x target labels.
* **Experiment:**
    * SageMaker DataWrangler to prepare data (joins), perform exploratory data analysis (EDA) and feature engineering of the feature columns.
    * SageMaker Autopilot to train and tune an optimal binary classifier.
* **Step by step instructions:**
    * Step by step instructions to execute this demo is documented [here](https://github.com/arunprsh/no-code-low-code/blob/main/finserv/tabular/README.md).
    
### Demo III - Language (SageMaker Jumpstart)

* **Use case: Sentiment Analysis for Financial News**
    * Investment firms have to sift through an overwhelming amount of information, largely in the form of unstructured text data, about sectors and companies of interest on a daily basis. Financial sentiment analysis is one of the essential components in navigating the attention over such continuous flow of data.
* **Dataset:**
    * Financial Phrase Bank - The dataset contains the sentiments for financial news headlines from the perspective of a retail investor. The dataset contains two columns, **label** (sentiment) and **headline**. The sentiment can be **negative**, **neutral** or **positive**.
* **Experiment:**
    * SageMaker Jumpstart to train a multi-class sentiment classifier.
    * Use a pre-trained TensorFlow **BERT Base Cased** model directly on the provided dataset for classification.
* **Step by step instructions:**
    * Step by step instructions to execute this demo is documented [here](https://github.com/arunprsh/no-code-low-code/blob/main/finserv/language/README.md).

### Demo IV - Vision (SageMaker Jumpstart)

* **Use case:** **Detecting Damaged Buildings using Satellite Images of Hurricane Damage**
    * After Hurricane Harvey struck Southeast Texas in September 2017, both Allstate and Farmers Insurance used drones to assess subsequent roof damage. The CEO of Farmers Insurance claimed that, through using drone images and computer vision (CV), the company reduced the time it took to assess hurricane-damage claims.
* **Dataset:**
    * The dataset comprises of satellite images from Texas after Hurricane Harvey divided into two groups (damage and no_damage). The goal is to make a computer vision model which can automatically identify if a given region is likely to contain flooding damage. The train set consists of 1000 images of each class.
* **Experiment:**
    * SageMaker Jumpstart to train an image classification model.
    * Fine-tune a pre-trained **ResNet18** CV model on provided dataset.
* **Step by step instructions:**
    * Step by step instructions to execute this demo is documented [here](https://github.com/arunprsh/no-code-low-code/blob/main/finserv/vision/README.md).