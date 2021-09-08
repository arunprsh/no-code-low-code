## Demo 3: SageMaker JumpStart (Language)

* *Text classification (pre-trained)*
    * For this demo, we will be using SageMaker Jumpstart to use one of its 100s of pre-trained language models directly for text-classification. 
    * You can also fine-tune these language models for various NLP downstream tasks like classification, summarization, question answering etc.
    * For this demo,  we will be using Electra-small++ pre-trained model to classify FinancialPhraseBank dataset. This dataset contains sentiments for financial news headlines. We will use Electra model to directly classify the headlines into 3 labels.
    * Navigate to the Text models in the Jumpstart view of models.
    * [Image: Screen Shot 2021-09-06 at 9.26.02 AM.png]
    * Clicking View all takes you to list of 106 text models that we currently support for NLP.
    * [Image: Screen Shot 2021-09-06 at 9.26.42 AM.png]
    * There are various models that support text classification, for this demo, let us pick BERT Base Cased. Type “text classification” in the search box and choose BERT Base Cased model for text classification (3rd in the results).
    * [Image: Screen Shot 2021-09-08 at 1.01.53 PM.png]
    * In the model view, choose the machine type you want for deploying. Remember, here we are not doing any fine-tuning, instead, we are directly deploying the pre-trained model as a SageMaker endpoint for real-time inference.
    * [Image: Screen Shot 2021-09-08 at 1.07.07 PM.png]
    * Click *Deploy*, this kicks off the deployment process and the view looks like the image below:
    * [Image: Screen Shot 2021-09-08 at 1.07.18 PM.png]
    * Next, it changes to the creation state as show below:
    * [Image: Screen Shot 2021-09-08 at 1.08.08 PM.png]
    * Once the deployment is done (as shown below), you can click the *Open Notebook* button to take a look at example code that shows how you can invoke this endpoint for classification.
    * [Image: Screen Shot 2021-09-08 at 1.20.08 PM.png]
    * The workshop repo contains a sample notebook which is a modified version of the above code example. It show cases how to use the deployed endpoint to perform sentiment classification. The notebook can be found here (https://github.com/arunprsh/no-code-low-code/blob/main/finserv/language/news-sentiment-classification/make-prediction.ipynb).

