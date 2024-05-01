![](UTA-DataScience-Logo.png)

# Kaggle Challenge: Metastatic Cancer Diagnosis Prediction

This repository holds an attempt to apply machine learning to metastatic cancer diagnosis predictions using data from the WiDS Datathon 2024 Challenge #1 Kaggle challenge that ran from 1/9/2024 - 3/1/2024.

*Kaggle Page Link: https://www.kaggle.com/competitions/widsdatathon2024-challenge1/overview.*

## Overview

The task, as defined on the Kaggle challenge page, is designed to help discover whether disparate treatments exist and to understand the drivers of biases, such as demographic and societal factors as well as establishing a benchmark for metastatic cancer diagnosis prediction. The approach in this repository formulates the problem as a binary classification task (cancer or no cancer), using machine learning models with categorical and numerical features as input. The task is to assess whether the likelihood of a patient’s diagnosis period being less than 90 days is predictable using characteristics and information about the patient from training and testing data. The overall goal is to determine whether a patient was diagnosed with metastatic cancer within 90 Days of screening. The primary goal of building these models is to detect relationships between demographics of a patient with the likelihood of getting timely treatment. The secondary goal is to see if environmental hazards impact proper diagnosis and treatment. For this challenge, I compared the performance of 3 different models. The best model was able to predict whether a patient was diagnosed with any cancer within 90 days of screening with a 100% accuracy. At the time of writing, the best performance on Kaggle of this metric is 18%.

## Summary of Workdone

### Data

* Data:
  * Context: The dataset is from 2015-2018 of patients who went to get screened for breast cancer. See Kaggle challenge main page for more information.
  * Type: 
    * Input: patient data (12,906 patients, , CSV file:  -> diagnosis
    * Input: CSV file of features, output: cancer/no cancer in last column.
  * Size: How much data?
  * Instances (Train, Test, Validation Split): how many data points? Ex: 1000 patients for training, 200 for testing, none for validation

#### Preprocessing / Clean up

* Exploratory data analysis, data preprocessing, data cleaning, data visualization, and feature engineering on training and test data

#### Data Visualization

Show a few visualization of the data and say a few words about what you see.

### Problem Formulation

* How can machine learning be leveraged to aid in cancer research and treatment?
  * Pretrain a ML model for that leverages AI & machine learning for cancer research, treatment, diagnosis, and prediction
  * Models
    * The 3 models I tested were XGBoost, Stochastic Gradient Descent, and Gradient Boosting.

### Training

  * Training was done in a Jupyter Notebook environment utlizing Ubuntu version 22.04 LTS. This was on a Dell Workstation Precision 5510 laptop.
  * Most of the training went smooth. However fine-tuning the selected model taking over 3+ hours.
  * Deciding on which training and test sets from the data was straightforward when looking at the ROC curves and measuring for AUC as well as cross-validating model accuracy.
  * After selecting the best training and test sets. Training the model was fairly straightforward.
  * Some difficulities I ran into were at Notebook #1 for data preprocessing and at Notebook #2 with fine tuning the model I selected. Be sure to see notes in the notebooks.
  * Some light hyperparamter fine-tuning was done. Though was crunched for time on project final submission.

### Performance Comparison

* Clearly define the key performance metric(s).
* Show/compare results in one table.
* Show one (or few) visualization(s) of results, for example ROC curves.
* The 3 models I trained, tested, cross-validated on the training data, the best performance metrics was the Gradient Boosting Model. I


### Conclusions

* The Gradient Boosting model works better than XGBoost and Stochastic Gradient Descent models. I would have liked to try out GPU training, testing, and fine-tuning to see if it is faster. Though I've read mostly TensorFlow utilizes GPU compute and also that Scikit-Learn libraries do not have support for GPU compute. Though I could be wrong about this. The training on my machine was done on CPU and RAM memory.

* In the little time window I had before submitting my project for this class, I was browsing through the Kaggle main challenge page. Other participants of the challenge I saw trained and tested over 20 models. The most popular model I noticed reading through the various approaches were CatBoost and utilizing Optuna for hyperparameter fine-tuning.

* Concluding thoughts, I believe in leveraging AI, machine learning, and at that deep learning for healthcare and better our everyday lives. This is essentially why I selected this challenge as a project for my data science computer programming course this semester.

### Future Work

* This machine learning project I hope can contribute to AI for healthcare, helping people, and society. I think for this project, the next thing I would do is look into why NO2 was so pronounced in my training sets. I came across an article about this from Harvard School of Public Health. See link: https://www.hsph.harvard.edu/news/press-releases/outdoor-air-pollution-may-increase-non-lung-cancer-risk-in-older-adults/ . In notebook 1 see Heatmap section. From a simple Google search, I found that the most prominent source of NO2 are internal combustion engines: cars. This may be worth looking into. Something else to look into is socioeconomic status. That feature came up as important in the model I selected for training. Another contributor found the metastatic cancer diagnosis codes to be the strongest predictive feature in their model. This should also be looked into as well. More future work also I would say is more exploratory data analysis on the data. I did not extensively look into drawing more insights from the data such as race/ethnicity, age, commute time, education, and location. For more context you can see other contributors notebooks on Kaggle about the data to give you more insight and context. I believe there are alot of studies and use cases for this specific challenge. Cancer, viruses - COVID-19 recently in 2020, public health studies. There are many use cases for this model.
  
## How to reproduce results

* To replicate my exact results:
   * Jupyter Notebooks: Do not run every cell in each notebook. For Notebook #1: import libraries and software packages, load and read in the training and test data, one-hot encode the categorical features with Pandas and Scikit-Learn, make sure to drop columns that are not needed, run the cells that take care of null values. You are doing this for the training data and the testing data. But if you want you can choose to one-hot encode once for the training data and once for the testing data. For Notebook #2: now that you have your training and testing data, notebook 2 is very straightforward. I tried to make it easy to understand. Notebook #2 has the final model I selected with the window of time that I had. Here you can really build from there. Notebook #1 on the other hand is for building robust data for training and testing.
   * Windows, Mac users, VS Code, etc: You can probably take bits and pieces of Notebook #1 and #2 for your usage purposes. There are many environments out there for machine learning. Some use Google Colab, Jupyter Notebooks, Ubuntu, VS Code. I noticed the hot trend in machine learning these days are VS Code and GPUs.

### Overview of files in repository

  * eda-preprocessing.ipynb: Notebook 1 for prepping data for ML training, testing, and validation.
  * ml-inference-submission.ipynb: Notebook 2 for ML model training, testing, validation, selection, and submission of model predictions.
  * README.md: Breakdown of Github repository and its files.
  * test.csv: original test data from Kaggle 
  * training.csv: original training data from Kaggle
  * sample_submission.csv: sample submission file for Kaggle challenge
  * pandas-test.csv: testing data from notebook 1 using Pandas one-hot encoding methods
  * pandas-train.csv: training data from notebook 1 using Pandas one-hot encoding methods
  * scikitlearn-test.csv: testing data from notebook 1 using Scikit-Learn's one-hot encoding methods
  * scikitlearn-train.csv: training data from notebook 1 using Scikit-Learn's one-hot encoding methods

  * Note: You can skip Notebook #1 and directly download the training and testing data from this repo and use Notebook #2 for ML model training, testing, exploration.


### Software Setup

* Software libraries and packages needed: Scikit-Learn, Numpy, Seaborn, Pandas, Matplotlib, Math, XGBoost, IPython, and tabulate.
* From the libraries you can import the specific packages listed at the top of each notebook that you will need. If your machine does not have it check online. Most if not all of them have documentation for installing on your machine.

### Data

* The original training and testing data can be downloaded from the Kaggle main challenge page. See link at the top. Browse over to data, and you can download them from there. The main idea in preprocessing the provided data is you are benchmarking numerical and categorical features in a way for binary classification of predicting cancer or no cancer.

### Training

* The most imporant thing I learned during this challenge was having the correct training and testing data. From there you divide up your training and testing data. Training data is for training and validating the model/s. Once you have decided on the most optimal model and best parameters, test your final model with the best parameters to make your final predictions.

#### Performance Evaluation

* For performance evaluation, you can run multiple models in 1 go or 1 at a time. I chose 3 in 1 go and selected the best I thought would give the best and most accurate prediction. Check the graphs and cross-validation accuracy scores to help on selecting a final model. Use hyperparameter tuning to fine-tune your final model for best results.

## Citations & Acknowledgements
* This Kaggle notebook was fairly helpful for me. Thanks Dee Dee. I also want to thank my professor Dr. Farbin and my graduate TA's Vineesha and Kunal for answering my questions throughout the semester.
* Kaggle contributors:
* - dee dee @ddosad Kaggle notebook: https://www.kaggle.com/code/ddosad/wids-data-exploration-ml-starter
* Harvard School of Public Health: https://www.hsph.harvard.edu/

 
*If you found this project helpful. Feel free to connect with me. Just ping me here on Github or my socials. Cheers.







