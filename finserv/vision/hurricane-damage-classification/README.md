## Hurricane Damage Classification

### Use case: Detecting Damaged Buildings using Satellite Images of Hurricane Damage
After Hurricane Harvey struck Southeast Texas in September 2017, both Allstate and Farmers Insurance used drones to assess subsequent roof damage. The CEO of Farmers Insurance claimed that, through using drone images and computer vision (CV), the company reduced the time it took to assess hurricane-damage claims.

### Dataset:
The dataset comprises of satellite images from Texas after Hurricane Harvey divided into two groups (damage and no_damage). The goal is to make a computer vision model which can automatically identify if a given region is likely to contain flooding damage. The train set consists of 1000 images of each class.

### Experiment:
SageMaker Jumpstart to train an image classification model.
Fine-tune a pre-trained `ResNet18` CV model on provided dataset.