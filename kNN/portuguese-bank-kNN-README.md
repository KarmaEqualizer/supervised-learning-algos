Here is a dataset on a direct marketing campaign run by a Portuguese bank and the output y is whether or not they subscribed to a term deposit. Note: the columns are delimited by semicolons rather than commas. Just pass delimiter=';' as a keyword argument in the call pandas.read_csv(.....) when you read the data in.

Here is a description of what each column means.

Part 1

Apply k-Nearest Neighbors on subsets of the columns and using cross validation see how the accuracy changes.
Now use all of the columns for your kNN classifier and see what the accuracy is.
Part 2

Use logistic regression on the data, perhaps bucketing some of the column data into discrete categories rather than being continuous values.
Try logistic regression on subsets of the columns.
Part 3

Between logistic regression and kNN, which performed better in predicting which customers would sign on for a term deposit?

http://nbviewer.ipython.org/gist/anandshudda/e4573e4c5f6bca2fd04c

From: https://github.com/suneel0101/datascience-2015/tree/master/lessons/05_k_nearest_neighbors

https://s3-us-west-2.amazonaws.com/ga-dat-2015-suneel/datasets/bank-additional-names.txt
