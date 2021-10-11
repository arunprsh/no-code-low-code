## Experimenting faster with No Code/Low Code ML tools from AWS

The Machine Learning (ML) journey requires continuous experimentation and rapid prototyping to be successful. In order to create highly accurate and performant models, data scientists have to first experiment with feature engineering, model selection and optimization techniques. These processes are traditionally time consuming and expensive. In this tech talk, data scientists will learn how the low-code / no-code capabilities found in **Amazon SageMaker Data Wrangler**, **Autopilot** and **Jumpstart**, make it easier to experiment faster and bring highly accurate models to production more quickly and efficiently.

* **What you will learn:**
    * Learn how to simplify the process of data preparation and feature engineering, and complete each step of the data preparation workflow.
    * Understand how to automatically build, train, and tune the best machine learning models based on your data, while allowing you to maintain full control and visibility.
    * Get started with ML easily and quickly using pre-built solutions for common use cases and open source models from popular model zoos.

## Representative Use Cases for Health Care and Life Sciences (HCLS) domain:
**Note:** If you are new to SageMaker Studio, instructions to get started with SageMaker Studio and setup this demo repository are provided [here](https://github.com/arunprsh/no-code-low-code/blob/main/setup-studio.md).

###  Demo I & II - Tabular (SageMaker DataWrangler + Autopilot)

* **Use case:** **Predict Diabetic Patients' Hospital Readmission**
    <p align="center"><img width="750" height="500" src="./img/image-1.jpeg"></p>
    * Identify the factors that lead to the high readmission rate of diabetic patients within 30 days post discharge and correspondingly predict the high-risk diabetic-patients who are most likely to get readmitted within 30 days. 
    * Hospital readmission is an important contributor to total medical expenditures and is an emerging indicator of quality of care. Diabetes, similar to other chronic medical conditions, is associated with increased risk of hospital readmission. hospital readmission is a high-priority health care quality measure and target for cost reduction, particularly within 30 days of discharge. The burden of diabetes among hospitalized patients is substantial, growing, and costly, and readmissions contribute a significant portion of this burden. Reducing readmission rates among patients with diabetes has the potential to greatly reduce health care costs while simultaneously improving care.
    * It is estimated that 9.3% of the population in the United States have diabetes , 28% of which are undiagnosed. The 30-day readmission rate of diabetic patients is 14.4 to 22.7 % . Estimates of readmission rates beyond 30 days after hospital discharge are even higher, with over 26 % of diabetic patients being readmitted within 3 months and 30 % within 1 year. Costs associated with the hospitalization of diabetic patients in the USA were `$124` billion, of which an estimated `$25` billion was attributable to 30-day readmissions assuming a 20 % readmission rate. Therefore, reducing 30-day readmissions of patients with diabetes has the potential to greatly reduce healthcare costs while simultaneously improving care.

* **Dataset:** 
    * The data set represents 10 years (1999-2008) of clinical care at 130 US hospitals and integrated delivery networks. It includes over 50 features representing patient and hospital outcomes.
    * The data contains such attributes as patient number, race, gender, age, admission type, time in hospital, medical specialty of admitting physician, number of lab test performed, HbA1c test result, diagnosis, number of medication, diabetic medications, number of outpatient, inpatient, and emergency visits in the year before the hospitalization, etc.
    * The data set contains 101,766 rows and 50 feature columns (including the target column).
    
| **Column name**       | **Description**     | 
| :------------- | :---------- | 
|`Encounter ID`|Unique identifier of an encounter|
|`Patient number`|Unique identifier of a patient|
|`Race Values`| Caucasian, Asian, African American, Hispanic, and other|
|`Gender Values`| Male, Female, and Unknown/Invalid|
|`Age Grouped in 10-year intervals`|[0-10), [10-20), ..., [90-100)|
|`Weight`| Weight in pounds|
|`Admission type`|Integer identifier corresponding to 9 distinct values, for example, emergency, urgent, elective, newborn, and not available|
|`Discharge disposition`|Integer identifier corresponding to 29 distinct values, for example, discharged to home, expired, and not available|
|`Admission source`|Integer identifier corresponding to 21 distinct values, for example, physician referral, emergency room, and transfer from a hospital|
|`Time in hospital`|Integer number of days between admission and discharge|
|`Payer code`|Integer identifier corresponding to 23 distinct values, for example, Blue Cross/Blue Shield, Medicare, and self-pay Medical<br>
|`Medical specialty`|Integer identifier of a specialty of the admitting physician, corresponding to 84 distinct values, for example, cardiology, internal medicine, family/general practice, and surgeon|
|`Number of lab procedures`|Number of lab tests performed during the encounter|
|`Number of procedures`|Numeric Number of procedures (other than lab tests) performed during the encounter|
|`Number of medications`|Number of distinct generic names administered during the encounter|
|`Number of outpatient visits`|Number of outpatient visits of the patient in the year preceding the encounter|
|`Number of emergency visits`|Number of emergency visits of the patient in the year preceding the encounter|
|`Number of inpatient visits`|Number of inpatient visits of the patient in the year preceding the encounter|
|`Diagnosis 1`|The primary diagnosis (coded as first three digits of ICD9); 848 distinct values|
|`Diagnosis 2`|Secondary diagnosis (coded as first three digits of ICD9); 923 distinct values|
|`Diagnosis 3`|Additional secondary diagnosis (coded as first three digits of ICD9); 954 distinct values|
|`Number of diagnoses`|Number of diagnoses entered to the system|
|`Glucose serum test result`|Indicates the range of the result or if the test was not taken. Values: ">200", ">300",  "normal" and "none" if not measured|
|`A1c test result`|Indicates the range of the result or if the test was not taken. Values: ">8" if the result was greater than 8%, ">7" if the result was greater than 7% but less than 8%, "normal" if the result was less than 7%, and "none" if not measured.|
|`Change of medications`|Indicates if there was a change in diabetic medications (either dosage or generic name). Values: "change" and "no change"|
|`Diabetes medications`|Indicates if there was any diabetic medication prescribed. Values: "yes" and "no" for 24 different kind of medical drugs.|
|`Readmitted`|Days to inpatient readmission. Values: "0" if the patient was readmitted in less than 30 days, ">30" if the patient was readmitted in more than 30 days, and "No" for no record of readmission|

* **Experiment:**
    * SageMaker DataWrangler to perform exploratory data analysis (EDA) and feature engineering on the feature columns.
    * SageMaker Autopilot to train and tune an optimal multi-class classifier.

* **Step by step instructions:**
    * Step by step instructions to execute this demo is documented [here]().
    
### Demo III - Language (SageMaker Jumpstart)

* **Use case:** **Classify medical specialties based on transcription text**
    <p align="center"><img width="750" height="500" src="./img/image-2.png"></p>
    * The medical subdomain (specialty) of a clinical note, such as cardiology or neurology, is a useful content-derived metadata for developing machine learning downstream applications. Clinical notes, in which the medical reports are mainly written in natural language, have been regarded as a powerful resource to solve different clinical questions by providing detailed patient conditions, the thinking process of clinical reasoning, and clinical inference, which usually cannot be obtained from the other components of the electronic health record (EHR) system (e.g., claims data or laboratory examinations). 
    * Automated document classification is generally helpful in further processing clinical documents to extract these kinds of data. As such, the massive generation of clinical notes and rapidly increasing adoption of EHR systems has caused automated document classification to become an important research field of clinical predictive analytics, to help leverage the utility of narrative clinical notes. In this demo, we will using state-of-the-art transformer-based language models like **BERT** pre-trained on **MEDLINE/PubMed** data to design a custom specialty classifier for our use case.
    * In order to learn the custom knowledge instilled by our corpus (transcripts), we fine-tune the BERT model with our data specifically for the downstream task of classification.

* **Dataset:**
    * This dataset contains sample medical transcriptions for various medical specialties.

* **Experiment:**
    * SageMaker Jumpstart to train a multi-class classifier.
    * Use a pre-trained TensorFlow **BERT** pre-trained on MEDLINE/PubMed** model directly on the provided dataset for classification.

* **Step by step instructions:**
    * Step by step instructions to execute this demo is documented [here](https://github.com/arunprsh/no-code-low-code/blob/main/hcls/language/README.md).

### Demo IV - Vision (SageMaker Jumpstart)

* **Use case:** **Detecting Damaged Buildings using Satellite Images of Hurricane Damage**
    <p align="center"><img width="500" height="500" src="./img/image-3.png"></p>
    * After Hurricane Harvey struck Southeast Texas in September 2017, both Allstate and Farmers Insurance used drones to assess subsequent roof damage. The CEO of Farmers Insurance claimed that, through using drone images and computer vision (CV), the company reduced the time it took to assess hurricane-damage claims.
    
* **Dataset:**
    * [YOLO medical mask dataset](https://www.kaggle.com/gooogr/yolo-medical-mask-dataset). The data set comprises of photos of people with medical masks, labeled for odject detection.
    
* **Experiment:**
    * SageMaker Jumpstart to train an image classification model.
    * Fine-tune a pre-trained **SSD MobileNet 1.0** CV model on the provided dataset. This is an object detection model from Gluon CV. It takes an image as input and returns bounding boxes for the objects in the image.
    * The model is pre-trained on **COCO 2017** dataset which comprises images with multiple objects and the task is to identify the objects and their positions in the image. A list of the objects that the model can identify is given at the end of the page.
    
* **Step by step instructions:**
    * Step by step instructions to execute this demo is documented [here](https://github.com/arunprsh/no-code-low-code/blob/main/hcls/vision/README.md).