## Low Code/No Code: Representative use cases for Financial Services

###  I.  Tabular (SageMaker DataWrangler + Autopilot)

* **Use case:** **Loan Default Analysis and Classification**
    * Lending loans to ‘risky’ applicants is the largest source of financial loss (called credit loss). The credit loss is the amount of money lost by the lender when the borrower refuses to pay or runs away with the money owed. In other words, borrowers who default cause the largest amount of loss to the lenders.
* **Dataset:** 
    * Lending Club Loan Dataset: The dataset contains complete loan data for all loans issued through the 2007–2011, including the current loan status and latest payment information.
    * 10000 rows, 21 feature columns
* **Experiment:**
    * SageMaker Autopilot to train and tune an optimal binary classifier.
    * SageMaker DataWrangler for exploratory data analysis (EDA) of feature columns.

### II. Vision (SageMaker Jumpstart)

* **Use case:** **Detecting Damaged Buildings using Satellite Images of Hurricane Damage**
    * After Hurricane Harvey struck Southeast Texas in September 2017, both Allstate and Farmers Insurance used drones to assess subsequent roof damage. The CEO of Farmers Insurance claimed that, through using drone images and computer vision (CV), the company reduced the time it took to assess hurricane-damage claims.
* **Dataset:**
    * The dataset comprises of satellite images from Texas after Hurricane Harvey divided into two groups (damage and no_damage). The goal is to make a computer vision model which can automatically identify if a given region is likely to contain flooding damage. The train set consists of 1000 images of each class.
* **Experiment:**
    * SageMaker Jumpstart to train an image classification model.
    * Fine-tune a pre-trained **ResNet18** CV model on provided dataset.

### III.  Language (SageMaker Jumpstart)

* **Use case: Sentiment Analysis for Financial News**
    * Investment firms have to sift through an overwhelming amount of information, largely in the form of unstructured text data, about sectors and companies of interest on a daily basis. Financial sentiment analysis is one of the essential components in navigating the attention over such continuous flow of data.

* **Dataset:**
    * Financial Phrase Bank - The dataset contains the sentiments for financial news headlines from the perspective of a retail investor. The dataset contains two columns, **sentiment** (label) and **headline**. The sentiment can be **negative**, **neutral** or **positive**.
* **Experiment:**
    * SageMaker Jumpstart to train a multi-class sentiment classifier.
    * Fine-tune a pre-trained **ELECTRA-Small++** model on provided dataset.