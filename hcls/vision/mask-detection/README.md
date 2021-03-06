### Demo IV - Vision (SageMaker Jumpstart)

* **Use case:** **Mask Detection**
    <p align="center"><img width="500" height="500" src="./img/image-3.png"></p>
    
    * There are many solutions to prevent the spread of the COVID-19 virus and one of the most effective solutions is wearing a face mask. Almost everyone is wearing face masks at all times in public places during the coronavirus pandemic. This encourages us to explore face mask detection technology to monitor people wearing masks in public places. Most recent and advanced face mask detection approaches are designed using deep learning. In this demo, we will see how to use state-of-the-art object detection models from SageMaker Jumpstart like SSD MobileNet, EfficientNet, VGG16, ResNet etc. to fine-tune an object detection for our data set.
    
* **Dataset:**
    * [YOLO medical mask dataset](https://www.kaggle.com/gooogr/yolo-medical-mask-dataset). The data set comprises of 75 photos of people (subset of original dataset) with medical masks, labeled for odject detection. The annotations are converted from the original YOLO to bbox format in compliance with Jumpstart requirements.
    
* **Experiment:**
    * SageMaker Jumpstart to train an image classification model.
    * Fine-tune a pre-trained **SSD MobileNet 1.0** CV model on the provided dataset. This is an object detection model from Gluon CV. It takes an image as input and returns bounding boxes for the objects in the image.
    * The model is pre-trained on **COCO 2017** dataset which comprises images with multiple objects and the task is to identify the objects and their positions in the image. A list of the objects that the model can identify is given at the end of the page.
    
* **Step by step instructions:**
    * Start with *explore-data.ipynb* example notebook to look at the training data we will be using for this exercise. Running the notebook also copies the training images to S3 from local `data` folder in this repo.
    * For this exercise, navigate to the vision models under SageMaker Jumpstart.
        <p align="center"><img width="500" height="500" src="./img/vision.png"></p>
        
    * Here, you can find over 200+ computer vision (CV) models pre-trained on open data for different downstream CV tasks. E.g., image classification, object detection, semantic segmentation etc. For this exercise, we want to fine-tune a CV model for object detection, hence, type "object detection" in the search box and choose SSD MobileNet 1.0 as the model to fine-tune.
        <p align="center"><img width="500" height="500" src="./img/object-detection.png"></p>
          
    * You can use the pre-trained model as it is or fine-tune it on your own custom images for the classification task. For our purpose, we will choose fine-tune.
        <p align="center"><img width="500" height="500" src="./img/SSD.png"></p>
        
    * Here, enter the S3 location to the input images and specifiy the name for the trained model alongside choosing the type of instance for training (fine-tuning). For hyperparameters, you can use the default.
        <p align="center"><img width="500" height="500" src="./img/fine-tune-SSD.png"></p>
        
    * Once the fine-tuning is complete, you can deploy the model as a real-time inference endpoint. 
    * Use the *make-prediction.ipynb* example notebook to explore the input images and upload the input data from local *data* folder here to S3.