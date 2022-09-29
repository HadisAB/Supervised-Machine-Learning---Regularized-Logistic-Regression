# Supervised-Machine-Learning---Regularized-Logistic-Regression
This model is the solution of one of the Supervised Machine learning projects in Coursera training. 
In this short project we are going to apply Regularized Logistic Regression model with full mathematical operations without using scikit learn.

## Project details
In this project, we will implement regularized logistic regression to predict whether microchips from a fabrication plant passes quality assurance (QA). During QA, each microchip goes through various tests to ensure it is functioning correctly.<br/>
We have the test results for some microchips on two different tests.

<ul>
<li>From these two tests, we would like to determine whether the microchips should be accepted or rejected.</li>
<li>y=1 means the microchips were accepted.</li>
<li>y=0 means the microchips were not accepted.</li>
</ul>


## Getting Started

1. Download [Anaconda](https://www.anaconda.com/distribution/) and install it.
2. Below packages need to be installed in your environment, too.

> [numpy](https://numpy.org/)<br/>
> [matplotlib.pyplot](https://matplotlib.org/) 

## Result
Following figure shows the training data in this project. It shows that our dataset cannot be separated into positive and negative examples by a straight-line through the plot. Therefore, a straight forward application of logistic regression will not perform well on this dataset since logistic regression will only be able to find a linear decision boundary.<br/>
<br/>
<img src=https://github.com/HadisAB/Supervised-Machine-Learning---Regularized-Logistic-Regression/blob/main/MicrochipTest.jpg />

<br/>
After applying the model into dataset you can find the decision boundary earned by Regulized logistic Regression and about 80% accuracy as below.<br/>

Click to find [scripts](https://github.com/HadisAB/Supervised-Machine-Learning---Regularized-Logistic-Regression/blob/main/Regularized_%20LogisticRegression_QA_%20microchips.py).
<br/>
<img src=https://github.com/HadisAB/Supervised-Machine-Learning---Regularized-Logistic-Regression/blob/main/MicrochipTest_DecisionBoundary.jpg />
