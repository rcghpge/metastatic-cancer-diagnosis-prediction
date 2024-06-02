<p align="center">
  <img src="UTA-DataScience-Logo.png" />
</p>

<!-- Use this div with the custom class around your text -->
<div class="custom-text">

## Kaggle Challenge: Metastatic Cancer Diagnosis Prediction. Building and Benchmarking ML Models

This repository holds an attempt to apply machine learning to metastatic cancer diagnosis predictions using data from the WiDS Datathon 2024 Kaggle Challenge #1 that ran from 1/9/2024 - 3/1/2024.

*Kaggle Page Link: https://www.kaggle.com/competitions/widsdatathon2024-challenge1/overview.*

## Overview

The Kaggle challenge aims to identify biases in metastatic cancer diagnosis, focusing on factors like demographics and societal issues. The task in this repository is to predict whether a patient's diagnosis occurs within 90 days, using a binary classification model with various categorical and numerical features.

The primary objective is to examine if the model can identify relationships between patient demographics and timely cancer diagnosis. A secondary objective is to explore the impact of environmental factors on diagnosis and treatment.

Three models were compared to achieve this. The top-performing models achieved accuracy rates of approximately 81% and 81.01%, respectively. Currently, the best performance on the Kaggle leaderboard for this challenge stands at 82.1%.

## Summary of Workdone

### Data

* Context: The dataset is from 2015-2018 of patients who went to get screened for breast cancer. See Kaggle page for more information.
  * Type:
    * Input: aggregate data (~12,906 patients) , CSV file: train.csv, test.csv -> diagnosis
    * Input: CSV file of patients and agreggate features, output: cancer/no cancer in last column.
  * Size: The original training and testing data was about 15MB. Including the training and testing data that I made, total training and testing data can be about ~80 MB.
  * Instances (Train, Test, Validation Split): I originally split the training and testing data into 2 categories: Pandas and Scikit-Learn. I ran into issues with data shape. This should be work for future work.

#### Preprocessing / Clean up

* Exploratory data analysis, data preprocessing, data cleaning, data visualization, and feature engineering on training and test data

#### Data Visualization
<p>
Benchmarking my two models was both challenging and insightful. During testing, I observed how two of my three chosen models were learning from the training and testing data. I didn't fine-tune extensively, which could have taken over 100 hours with the parameters I considered. Here are some of the initial insights from the two models.
</p>

##### XGBoost Learning Curve
<div><img src="https://github.com/rcghpge/metastatic-cancer-diagnosis-prediction/blob/main/images/XGBoost%20Model%20Learning%20Curve.png" with="450" height="450"></div>


##### Gradient Boosting Learning Curve
<div>
<img src="https://github.com/rcghpge/metastatic-cancer-diagnosis-prediction/blob/main/images/GBD%20Model%20Learning%20Curve.png" width="500" height="500">
</div>

### Problem Formulation

* How can machine learning be leveraged to aid in cancer research and treatment?
  * Pretrain a ML model that leverages AI & machine learning for cancer research, treatment, diagnosis, and prediction
  * Models
    * The 3 models I tested were XGBoost, Stochastic Gradient Descent, and Gradient Boosting. I settled on 2 final models.

### Training

* Training took place in a Jupyter Notebook on Ubuntu 22.04 LTS, using a Dell Workstation Precision 5510 laptop.
* Most training went smoothly, but fine-tuning the selected model took over three hours.
* The training and test sets were straightforward to decide on the most optimal model by examining ROC curves and measuring AUC, while also cross-validating for model accuracy. However, I encountered issues with data shape.
* The main challenges were preprocessing, fitting, and fine-tuning.
* Some light hyperparameter fine-tuning was attempted, but I was pressed for time due to the project deadline.

### Performance Comparison

* All 3 models performed fairly well. The best performing model seemed to be Gradient Boosting. You can see below from the ROC curve graph. I wasn't happy with 81.01% accuracy and because of my issues with data shape, I believe this affected the peformance of the models. If I had more time on the project, I would have pushed for 85% to over 90% accuracy.

#### ROC Curve Graph

<p><div><img src="https://github.com/rcghpge/metastatic-cancer-diagnosis-prediction/blob/main/images/Pandas%20Training%20Data%20ROC%20AUC%20Curve.png" width="600" height="500"></div></p>

### Conclusions

* The Gradient Boosting model worked the best for me compared to the XGBoost and Stochastic Gradient Descent models. I would have liked to try out GPU training, testing, and fine-tuning to see if it is faster. Though I've read mostly TensorFlow utilizes GPU compute and also that Scikit-Learn libraries do not have support for GPU compute. Though I just getting into machine learning. The training on my machine was done on CPU and RAM memory.

* In the little time window I had before submitting my project for this class, I was browsing through the Kaggle challenge page. Other participants of the challenge I saw trained and tested over 20 models. The most popular model I noticed reading through the various approaches were CatBoost and utilizing Optuna for fine-tuning.

* My concluding thoughts are that there is a lot of work that can be done from this challenge. This project alone could stand for benchmarking. Hopefully this will be of some use to someone.

### Future Work

* The next step is to investigate why NO2 (nitrogen dioxide) appears prominently in my training data (see the Heatmap section of Notebook #1). A Google search suggests that internal combustion engines, like those in motor vehicles, are a major source of NO2, indicating this could be worth exploring.

* Socioeconomic status also emerged as a significant feature during training, testing, and fine-tuning, suggesting it could influence outcomes. Additionally, another contributor found that metastatic cancer diagnosis codes were the strongest predictive feature, which should be further analyzed.

* An unexpected feature that gained importance during fine-tuning was geographical location. I'm not entirely sure why, but it might be linked to environmental or regional factors.

* While I haven't deeply analyzed the aggregate data, these various predictive features suggest there are many angles to explore for machine learning benchmarking. If you need more insights, you can refer to other contributors' notebooks on Kaggle. This challenge has can extend research in medicine, public health, among many other fields of study, illustrating the broad applicability of these models.

## How to reproduce results

To replicate my results:

 * **Jupyter Notebooks**: Do not run every cell in each notebook. For **Notebook #1**: import the necessary libraries, load the training and testing data, one-hot encode categorical features, drop unnecessary columns, and handle null values. This applies to both the training and testing data, though you can choose to one-hot encode once for each dataset.

 * In **Notebook #2**, I trained, fitted, and fine-tuned the original model using partial data, mainly from Scikit-Learn's training set. It can serve as a basic benchmarking model.

 * **Notebook #3** focuses on training, fitting, and fine-tuning Model 2 using the complete training data. This notebook can be used to evaluate the performance of the model. However, I encountered issues with data shape during this process.

 * **Windows, Mac, and Other Environments**: You can adapt the content from Notebooks #1, #2, and #3 for your setup, whether you're using Google Colab, Jupyter Notebooks, Ubuntu, or VS Code. Different machine learning environments offer various features; recent trends favor VS Code and GPUs for enhanced performance.

### Overview of files in repository

  * The repository has 2 folders images and notebooks. The notebooks folder contains the 3 notebooks below.
  * eda-preprocessing.ipynb: Notebook 1 for prepping aggregate data for ML training, fitting, testing, validation, and fine-tuning.
  * ml-inference-submission.ipynb: Notebook 2 for ML model training, testing, validation, selection, and submission of model predictions. Original model - Model 1.
  * ml-inference-submission2.ipynb: Notebook 3 for ML model training, testing, validation, selection, and submission of model predictions. Model 2.
  * README.md: Breakdown of Github repository and files.
  

  * Note: You can skip Notebook #1 and directly download the training and testing data from Kaggle and use Notebook #2 and #3 for model training, testing, exploration.


### Software Setup

* Software libraries and packages needed: Scikit-Learn, Numpy, Seaborn, Pandas, Matplotlib, Math, XGBoost, IPython, and tabulate.
* From the libraries you can import the specific packages listed at the top of each notebook that you will need. If your machine does not have it check online. Most if not all of them have documentation for installing on your machine.

* I came across a library called Imbalance Learn while I was preproccesing the training and testing data. Its a library for dealing with imbalance in datasets.
* See link: https://imbalanced-learn.org/stable/

### Data

* The original training and testing data can be downloaded from the Kaggle link above. Browse over to data, and you can download them from there. The main idea in preprocessing the data is that you are benchmarking numerical and categorical features for robust predictive binary classification in machine learning.

### Training

* The most imporant thing I learned during this challenge was having the correct training and testing data. From there you divide up your training and testing data. Training data is for training and validating the models. Once you have decided on the most optimal model and best parameters, fit, test, and fine-tune your final model with the best parameters to make your best predictions.

#### Performance Evaluation

* For performance evaluation, you can run multiple models in 1 go or 1 at a time. I chose 3 in 1 go and ended with the 2 best models I thought would give the best and most accurate prediction. Check the graphs and cross-validation accuracy scores to help on selecting a final model. Fine-tune your models for best results.

## Citations & Acknowledgements
* This Kaggle notebook was fairly helpful for me. Thanks Dee Dee. I also want to thank my professor Dr. Farbin and my graduate TA's Vineesha and Kunal for answering my questions throughout the semester.
* Kaggle challenge contributors:
* dee dee @ddosad Kaggle notebook: https://www.kaggle.com/code/ddosad/wids-data-exploration-ml-starter

 
If you found this project helpful. Feel free to connect with me. Just ping me here on Github or on socials. Cheers =)

</div>






