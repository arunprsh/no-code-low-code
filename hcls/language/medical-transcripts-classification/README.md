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
    * Use a pre-trained TensorFlow **BERT** pre-trained on **MEDLINE/PubMed** model directly on the provided dataset for classification.

* **Step by step instructions:**
    * Start by executing the *explore-data.ipynb* notebook in this repo to take a look at the data and specialities to be predicted. Running the notebook also uploads the data self contained in this repository to S3 for training later.
    * From the left icon pane, navigate to the Jumpstart icon and hit **Browse Jumpstart** as shown in the image below. As you can see, there are over 100+ language models pre-trained for various downstream tasks available within Jumpstart. Click view all models.
        <p align="center"><img width="950" height="700" src="./img/text-models.png"></p>
        
    * In the search box, you can type "text classification" to filter all the models that are trained specifically for text classification. For our use case of classifying medical transcripts to specialities, let us fine-tune a BERT model pre-trained on MEDLINE/PubMed data. Type "pubmed" to get to the model quickly.
        <p align="center"><img width="950" height="700" src="./img/pubmed.png"></p>
   
    * Once we click on the appropiate model we want to use for our use case, you can see 2 options - i) you can use the pre-trained model as it is to do the classification or ii) fine-tune the pre-trained model to learn the custom knowledge from our dataset.
        <p align="center"><img width="950" height="700" src="./img/fine-tune-1.png"></p>
        
    * Choose **Fine-tune Model** and enter basic information like s3 input path, model name and choose the instance type to be used for training. It is recommended to use a 'P' type instance with GPUs for training. For hyperparameters, we can use the defaults. 
        <p align="center"><img width="950" height="700" src="./img/fine-tune-2.png"></p>
    * Once the fine-tuning is complete, you can deploy the model for real-time inference.
    * The example notebook *make-prediction.ipynb* contained in this repo demonstrates how to invoke the deployed endpoint for classification (inference).