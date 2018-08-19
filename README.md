# Cotton-Procurement-Planner-using-ML
A cotton procurement planner  which predicts the sales based on historical sales data and also takes Marketing Team forecast as input using Machine Learning

Project Description :

I have built Cotton Procurement Planner System using Multi-Layer Neural Network which helps the mill to forecast the demand of cotton varieties by predicting the requirement based on historical sales data of the mill.

It includes the back end of the application Cotton Procurement System. It forecasts the demand of cotton variety. The machine learning algorithm of Multi-Layer Neural Network is used in this application and the Front-end application is a mobile application

Proposed System:

Programming Language: Python using Scikit Learn Libraries used : Numpy, Pandas

The Neural Network consists of three layers: 1. Input - Monthly sales data of 10 products .

Output – Prediction for next three months. 3.Hidden – Back propagation for reducing error in prediction and to increase accuracy with 3 neurons.

Data sets: We wanted to practically implement the problem statement.So using the data set that had been acquired from the mill in Madurai, Tamil Nadu.

Preprocessing of dataset:

We interpolated 3-year sales data into 36-month sales data using Cubic Spline Interpolation System. The missing data were filled using Nan median module.

The hidden layer is designed and it is biased in such a way to reduce the deviation of results from the actual data through backpropagation. The hidden layer consists of three neurons.

The activation function chosen is all that is the rectilinear unit function because it provided the best and accurate results. The metrics for each and every activation function is provided below:

relu: explained variance score- 0.9993743251047639, MAE- 10.621257584237828, MLSE- 0.6675717149427682, R2- 0.7247263290629995

sigmoid: explained variance score- 0.6972986389853218, MAE- 8.8253916239100601, MLSE- 0.8613298901237013, R2- 0.5682130912730017

tanh: explained variance score- 0.7961230721375639, MAE- 9.6125387983465824, MLSE- 0.8717269754914278, R2- 0.5986298115234268.
