# kaggle-Housing-Prices
STEPS IN SOLVING:->


1.
Import Libraries and data

2.
Define Median Absolute Deviation Function

3.
Remove Skew from SalesPrice data

4.
Merge Train and Test to evaluate ranges and missing values¶
This was done primarily to ensure that Categorical data in the training and testing data sets were consistent.

5.
Create separte datasets for Continuous vs Categorical
Creating two data sets allowed me to handle the data in more appropriate ways.

6.
Handle Missing Data for continuous data
If any column contains more than 50 entries of missing data, drop the column
If any column contains fewer that 50 entries of missing data, replace those missing values with the median for that column
Remove outliers using Median Absolute Deviation
Calculate skewness for each variable and if greater than 0.75 transform it
Apply the sklearn.Normalizer to each column

7.
Handle Missing Data for Categorical Data
If any column contains more than 50 entries of missing data, drop the column
If any column contains fewer that 50 entries of missing data, replace those values with the 'MIA'
Apply the sklearn.LabelEncoder
For each categorical variable determine the number of unique values and for each, create a new column that is binary

8.
Create Estimator and Apply Cross Validation
We can gauge the accuracy of our model by implementing an multi-fold cross validation and outputting the score. In this case I chose to run 15 iterations and output the score as Root Mean Squared Error.

The results range from ~0.11-0.17 with a mean of ~0.14.

9.
Evaluate Feature Significance
Investigating feature importance is a relatively straight forward process:

Out feature importance coefficients
Map coefficients to their feature name
Sort features in descending order
Given our choice of model and methods for preprocessing data the most significant features are:

OverallQual
GrLivArea
TotalBsmtSF
GarageArea

10.
Visualize Predicted vs. Actual Sales Price
In order to visualize our predicted values vs our actual values we need to split our data into training and testing data sets. This can easily be accomplished using sklearn's train_test_split module.

We will train the model using a random sampling of our data set and then compare visually against the actual values.
