
# Inventory Monitoring at Distribution Centers

**Distribution centers** often use robots to move objects as a part of their operations. Objects are carried in bins which can contain multiple objects.

In this project, we will have to build a model that can count the number of objects in each bin. A system like this can be used to track inventory and make sure that delivery consignments have the correct number of items.

To build this project we will use **AWS SageMaker** and good machine learning engineering practices to fetch data from a database, preprocess it, and then train a machine learning model. This project will serve as a demonstration of end-to-end machine learning engineering skills that we have learned as a part of this nanodegree.

## Project Set Up and Installation

Dependencies

````python
Python 3.7
PyTorch >=3.6
````

Installation

````python
pip install -r requirements.txt
````

For this project, it is highly recommended to use Sagemaker Studio from the course provided **AWS workspace**.

For local development, you will need to setup a jupyter lab instance.

Follow the [jupyter install](https://jupyter.org/install.html) link for best practices to install and start a jupyter lab instance.
If you have a python virtual environment already installed you can just pip install it.

````python
pip install jupyterlab
````

In **AWS Sagemaker**, we have created a bucket to store all of our downloaded images.

<figure>
  <img src="./fig/03.png" alt=".." title="Optional title" width="95%" height="70%"/>
</figure>

**OPTIONAL:** If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to make your `README` detailed and self-explanatory. For instance, here you could explain how to set up your project in AWS and provide helpful screenshots of the process.

## Dataset

### Overview

To complete this project we will be using the Amazon Bin Image Dataset. The dataset contains 500,000 images of bins containing one or more objects. Dataset can be access from [here](https://registry.opendata.aws/amazon-bin-imagery/)

For each image there is a metadata file containing information about the image like the number of objects, it's dimension and the type of object. For this task, we will try to classify the number of objects in each bin.

## Access

### 1. Resource type

````sh
S3 Bucket
````

### 2. Amazon Resource Name (ARN)

````sh
arn:aws:s3:::aft-vbi-pds
````

### 3. AWS Region

````sh
us-east-1
````

### 4.AWS CLI Access (No AWS account required)

````sh
aws s3 ls --no-sign-request s3://aft-vbi-pds/
````

Dataset is imbalance , here's a dataset distribution.

<figure>
  <img src="./fig/05.png" alt=".." title="Optional title" width="75%" height="70%"/>
</figure>

### 5. AWS Sagemaker Studio

* Launch `AWS Sagemaker Studio`.
* Import jupyter notebook file `sagemaker.ipynb`
* Make sure kernel is started.
* Run cells.

## Model Training

For this experiment, we have use a [Resnet50](https://viso.ai/deep-learning/resnet-residual-neural-network/) model and decide to tune some parameters:

````python
hyperparameter_ranges = {
    "learning_rate": ContinuousParameter(0.001, 0.1),
    "batch_size": CategoricalParameter([32, 64, 128, 256]),
    "epochs": CategoricalParameter([10,15, 25 , 30 ])    
}
````

Best hyperpapameters and jobs

<figure>
  <img src="./fig/06_.png" alt=".." title="Optional title" width="95%" height="70%"/>
</figure>

Best_estimator.hyperparameters

````python
{'_tuning_objective_metric': '"Test Loss"',
 'batch_size': '"32"',
 'epochs': '"25"',
 'learning_rate': '0.0012128565992639416',
 'sagemaker_container_log_level': '20',
 'sagemaker_estimator_class_name': '"PyTorch"',
 'sagemaker_estimator_module': '"sagemaker.pytorch.estimator"',
 'sagemaker_job_name': '"Capstone-awsmle-hpo-job-2022-05-26-07-05-00-441"',
 'sagemaker_program': '"tuner.py"',
 'sagemaker_region': '"us-east-1"',
 'sagemaker_submit_directory': '"s3://sagemaker-us-east-1-310754713715/Capstone-awsmle-hpo-job-2022-05-26-07-05-00-441/source/sourcedir.tar.gz"'}
 ````

## Model Profiling and Debugging

We've used model debugging and profiling to better monitor and debug your model training job. After that we generate a Profil report.

<figure>
  <img src="./fig/07.png" alt=".." title="Optional title" width="95%" height="70%"/>
  <img src="./fig/07_.png" alt=".." title="Optional title" width="95%" height="70%"/>
</figure>

## Model Evaluation

<figure>
  <img src="./fig/08.png" alt=".." title="Optional title" width="95%" height="70%"/>
</figure>

The accuracy of the [benchmark model](https://ieeexplore.ieee.org/document/8010578) chosen is 56 % (Approx), the experiment model didn’t achieve the results of the Benchmark. So as next steps, must work more on the data and its transformation. Maybe doing some data augmentation or transfer learning.

<figure>
  <img src="./fig/09.png" alt=".." title="Optional title" width="40%" height="70%"/>
  <img src="./fig/10.png" alt=".." title="Optional title" width="42%" height="70%"/>
</figure>

## Model Querying

* Open an image

````python

from PIL import Image
img_dict={ "url": "https://aft-vbi-pds.s3.amazonaws.com/bin-images/109.jpg" }
img_bytes = requests.get(img_dict['url']).content
Image.open(io.BytesIO(img_bytes))
````

* Inference

````python
response=predictor.predict(json.dumps(img_dict), initial_args={"ContentType": "application/json"})
response[0]
````

* Find the class

````python
np.argmax(response, 1)
````

## Furthermore

* Train oour model on multiple instances.
* Train our model on spot instances.

  We have seen a notable difference between:
  
  > **X** (the actual compute-time your training job spent) and

  > **Y** (the time you will be billed for after Spot discounting is applied) 
  
  signifying the cost savings you will get for having chosen Managed Spot Training.
This should be reflected in an additional line:

	Managed Spot Training savings:

  $$(1 - \frac{Y}{X})*100\%$$
