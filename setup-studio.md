## SageMaker Studio Setup Instructions 

*Note:* You need to use your own AWS Account to perform these steps. This may incur some charges. Make sure to cleanup all resources that are created as part of this workshop after your data science experimentations.

### Launch/Setup Amazon SageMaker Studio

Amazon SageMaker Studio is a web-based, integrated development environment (IDE) for machine learning that lets you build, train, debug, deploy, and monitor your machine learning models. Studio provides all the tools you need to take your models from experimentation to production while boosting your productivity.

Follow the steps below to onboard to Amazon SageMaker Studio using Quick Start. 

*Note:* If you already have a SageMaker Studio domain setup in your account, ensure it is updated to the latest version as per the instructions here (https://docs.aws.amazon.com/sagemaker/latest/dg/studio-tasks-update.html) and skip the below steps.

* Open AWS console and switch to AWS region you  would like to use.
    * SageMaker Studio is available in the following regions (https://docs.aws.amazon.com/sagemaker/latest/dg/studio.html).
* In the search bar, type *SageMaker* and  click on *Amazon SageMaker*.
    ![image](./img/image-1.png)
* Click on *Amazon SageMaker Studio* (first  option on the left pane).
    ![image](./img/image-2.png)
* Click on *Quick start*.
* Define *User name* as *sagemakeruser*  for example.
* Select *Create a new role* under  Execution role.
    ![image](./img/image-3.png)
* Keep the defaults and click *Create Role*.
    ![image](./img/image-4.png)
* You will see that the role is successfully  created.
    ![image](./img/image-5.png)

You can see that the option *Enable Amazon SageMaker project templates and JumpStart for this account and Studio users* is enabled. Keep this default setting.

* Choose the newly created role and click *Submit*.
    ![image](./img/image-6.png)
* The SageMaker Studio environment will stay in *Pending*  state for a few minutes.
    ![image](./img/image-7.png)
* After a few minutes, the state will transition  to *Ready*.
    ![image](./img/image-8.png)
* Once Amazon SageMaker Studio is ready then  click on *Open Studio*. The page can take 1 or 2 minutes to load when  you access SageMaker Studio for the first time.
    ![image](./img/image-9.png)
* You will be redirected to a new web tab that  looks like this:
    ![image](./img/image-10.png)

**Congratulations!** You have successfully created a SageMaker Studio domain.
 

Downloading the content of the GitHub repository needed for the workshop 

* In the SageMaker Studio *File*  menu, select **New** and then click on* *Terminal**.
    ![image](./img/image-11.png)
* In the terminal, type the following command:
    * $ git clone https://github.com/arunprsh/no-code-low-code.git
* After cloning, you will have *no-code-low-code* folder created in *left panel* of the  studio.

**Congratulations!** You have successfully downloaded the content of the *no-code-low-code* workshop.


**Note:** For the workshop, we recommend pre-initializing SageMaker Data Wrangler (follow steps below) after you setup SageMaker Studio as per the above instructions. 

 * Initialize SageMaker Data Wrangler via SageMaker Studio UI.
    ![image](./img/image-12.png)
    * Takes a few minutes to load.
    ![image](./img/image-13.png)
    * Once Data Wrangler is loaded, you should be able to see it under running instances and apps as shown below.
    ![image](./img/image-14.png)
    