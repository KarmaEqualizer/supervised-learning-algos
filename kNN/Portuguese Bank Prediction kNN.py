
# coding: utf-8

# In[75]:

import pandas as pd
import pylab as pl
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing


# In[76]:

## Read the data
df = pd.read_csv('/home/vagrant/repos/datasets/bank-additional-full.csv', delimiter=";")
## print df.keys()
## df.describe()


# In[77]:

## Select the feature set for first iteration
features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'y']
x_features = ['job', 'marital', 'education', 'default', 'housing', 'loan']

df_feats = df[features]


# In[78]:

""" 

## One Way of converting features to categorical variables:

job_types = ["admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown"]
marital_types = ["divorced","married","single","unknown"]
education_types = ["basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown"]
default_types = ["no","yes","unknown"]
housing_types = ["no","yes","unknown"]
loan_types = ["no","yes","unknown"]

types_dict = { "job":job_types, "marital":marital_types, "education":education_types, "default":default_types, "housing":housing_types, "loan":loan_types}

def convert_to_num(val, f):
   types = types_dict[f]
   return types.index(val)

for f in features:
    df_feats[f] = df_feats[f].apply(convert_to_num, args=(f,))

"""
print "Above method is just for practice"


# In[79]:

JOB_MAP = {
"admin." : 2,
"blue-collar" : 1, 
"entrepreneur" : 1, 
"housemaid" : 1, 
"management" : 3, 
"retired" : 0, 
"self-employed": 0, 
"services" : 2, 
"student" : 0, 
"technician" : 3, 
"unemployed" : 0, 
"unknown" : 0, }

df_feats["job"] = df_feats["job"].apply(lambda value: JOB_MAP.get(value))

MARITAL_MAP = {
"divorced" : 1, 
"married" : 2, 
"single" : 1, 
"unknown" : 0 }

df_feats["marital"] = df_feats["marital"].apply(lambda value: MARITAL_MAP.get(value))

EDUCATIONAL_MAP = {
"basic.4y" : 1, 
"basic.6y" : 2, 
"basic.9y" : 3, 
"high.school" : 0, 
"illiterate" : 0, 
"professional.course" : 1, 
"university.degree" : 1, 
"unknown" : 0 }

df_feats["education"] = df_feats["education"].apply(lambda value: EDUCATIONAL_MAP.get(value))

DEFAULT_MAP = {
"no" : 1, 
"yes" : 2, 
"unknown" : 0 }

df_feats["default"] = df_feats["default"].apply(lambda value: DEFAULT_MAP.get(value))

HOUSING_MAP = {
"no" : 1, 
"yes" : 2, 
"unknown" : 0 }

df_feats["housing"] = df_feats["housing"].apply(lambda value: HOUSING_MAP.get(value))

LOAN_MAP = {
"no" : 1, 
"yes" : 2, 
"unknown" : 0 }

df_feats["loan"] = df_feats["loan"].apply(lambda value: LOAN_MAP.get(value))

Y_MAP = {
"no" : 0, 
"yes" : 1 }

df_feats["y"] = df_feats["y"].apply(lambda value: Y_MAP.get(value))


# In[80]:

df_feats.head()


# In[81]:

## Creating test/training sets

test_idx = np.random.uniform(0, 1, len(df_feats)) <= 0.3

# The training set will be ~30% of the data
train = df_feats[test_idx==True]

# The test set will be the remaining, ~70% of the data
test = df_feats[test_idx==False]


# In[82]:

## kNN Classifier

results = []
# range(1, 51, 2) = [1, 3, 5, 7, ...., 49]
for n in range(1, 51, 2):
    clf = KNeighborsClassifier(n_neighbors=n)
    # train the classifier
    clf.fit(train[x_features], train['y'])
    # then make the predictions
    preds = clf.predict(test[x_features])
    
    ## finding the optimal value of k
    accuracy = np.where(preds==test['y'], 1, 0).sum() / float(len(test)) ## 
    print "Neighbors: %d, Accuracy: %3f" % (n, accuracy)

    results.append([n, accuracy])

results = pd.DataFrame(results, columns=["n", "accuracy"])

pl.plot(results.n, results.accuracy)
pl.title("Accuracy with Increasing K")
pl.show()



# In[83]:

print """Based on the above information, selecting k=9 for the further analysis"""

## Selecting the right predictor function

results = []

for w in ['uniform', 'distance']:
    clf = KNeighborsClassifier(9, weights=w)
    w = str(w)
    clf.fit(train[x_features], train['y'])
    preds = clf.predict(test[x_features])

    accuracy = np.where(preds==test['y'], 1, 0).sum() / float(len(test))
    
    print "Weights: %s, Accuracy: %3f" % (w, accuracy)

    results.append([w, accuracy])

results = pd.DataFrame(results, columns=["weight_method", "accuracy"])
print results, '\n'



# In[84]:

print """Based on the above results, our optimal combination is: 
k = 9 and 
classifier = distance"""


# In[85]:

""" Try the above analysis using logistic regression and measure accuracy """

import statsmodels.api as sm

## adding intercept
df_feats['intercept'] = 1.0

## logistic regression model
logit = sm.Logit(train["y"], train[x_features])
result = logit.fit()

print result.summary()

logit_preds = result.predict( test[x_features] )


# In[86]:

accuracy = np.where((logit_preds+0.5) >= test['y'], 1, 0).sum() / float(len(test))


# In[87]:

print "Accuracy of logistic regression for the above dataset is: %3f" % (accuracy)


# In[88]:

print "Apparently Logistic Regression is performing very similar if not a littler better than kNN(9, distance) for this dataset"


# In[ ]:



