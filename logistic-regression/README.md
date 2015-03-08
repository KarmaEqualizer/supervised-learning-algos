We'll be using the same dataset as UCLA's Logit Regression in R tutorial to explore logistic regression in Python. Our goal will be to identify the various factors that may influence admission into graduate school.

The dataset contains several columns which we can use as predictor variables:

gpa
gre score
rank or prestige of an applicant's undergraduate alma mater
The fourth column, admit, is our binary target variable. It indicates whether or not a candidate was admitted our not.

Key concepts:
 - Building a logit model using the statsmodel library
 - dummying variables for logistic regression
 - Using Pandas for data wrangling
 - Calculating Accuracy of the model using(test/training dataset) [% Accuracy: 68.75%]