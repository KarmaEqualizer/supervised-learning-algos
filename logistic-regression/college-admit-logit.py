## Split dat ainto training and validation sets

def get_training_and_validation_sets(feature_sets):

    # randomly shuffle the feature sets
    #random.shuffle(feature_sets)

    # get the number of data points that we have
    count = len(feature_sets)
    # 20% of the set, also called "corpus", should be training, as a rule of thumb, but not gospel.

    # we'll slice this list 20% the way through
    slicing_point = int(.20 * count)

    # the training set will be the first segment
    pct_20_set = feature_sets[:slicing_point]

    # the validation set will be the second segment
    pct_80_set = feature_sets[slicing_point:]
    
    return pct_20_set, pct_80_set

import csv
import numpy as np
import pandas as pd
import pylab as pl
import statsmodels.api as sm

# read and in the data and show summary statistics
data = pd.read_csv('../../datasets/admission_data.csv')
print "\n****DESCRIPTION OF THE DATA****"
print data.describe()


# show histogram of the data
data.hist()
pl.show()

# Split Data into training and validation
validation_data, training_data = get_training_and_validation_sets(data)

# this is just pandas notation to get columns 1...n
# we want to do this because our input variables are in columns 1...n
# while our target is in column 1 (0=not admitted, 1=admitted)
training_columns = training_data.columns[1:]
logit = sm.Logit(training_data["admit"], training_data[training_columns])
result = logit.fit()
print result.summary()

def predict(gre, gpa, prestige):
    """
    Outputs predicted probability of admission to graduate program
    given gre, gpa and prestige of the institution where
    the student did their undergraduate
    """
    #return result.predict([gre, gpa, prestige])[0]
    return result.predict([gre, gpa, prestige])


print "\nPrediction for GRE: 400, GPA: 3.59, and Tier 3 Undergraduate degree is..."
print predict(400, 3.59, 3)

dummy_ranks = pd.get_dummies(data['prestige'], prefix='prestige')

cols_to_keep = ['admit', 'gre', 'gpa']
input_data = data[cols_to_keep].join(dummy_ranks.ix[:, 'prestige_2':])

## Add intercept
input_data['intercept'] = 1.0

print input_data.head(20)

# Split Data into training and validation
new_validation_data, new_training_data = get_training_and_validation_sets(input_data)

# this is just pandas notation to get columns 1...n
# we want to do this because our input variables are in columns 1...n
# while our target is in column 1 (0=not admitted, 1=admitted)
training_columns = input_data.columns[1:]
logit = sm.Logit(new_training_data["admit"], new_training_data[training_columns])
result = logit.fit()
print result.summary()

def predict(gre, gpa, prestige_2, prestige_3, prestige_4 ):
    """
    Outputs predicted probability of admission to graduate program
    given gre, gpa and prestige of the institution where
    the student did their undergraduate
    """
    return result.predict([gre, gpa, prestige_2, prestige_3, prestige_4, 1])

    #return result.predict([gre, gpa, prestige_2, prestige_3, prestige_4, 1])[0]

print "\nPrediction for GRE: 400, GPA: 3.59, and Tier 1 Undergraduate degree is..."
print predict(400, 3.59, 0, 0, 0)

## Test the model using validation data 
new_results = result.predict( new_validation_data[training_columns] )

check = pd.DataFrame(new_validation_data["admit"])

check["actuals"] = new_results

check["admit_actuals"] = new_results

check.ix[check.actuals >= .5,'admit_actuals'] = 1
check.ix[check.actuals < .5,'admit_actuals'] = 0

check["accuracy"] = 0

check.ix[check.admit == check.admit_actuals,'accuracy'] = 1

print "% Accuracy: ", float(check['accuracy'].sum()) / float(check.count().accuracy)