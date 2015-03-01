Objective: Create a Naive Bayes Classifier that accurately classifies SMSs as spam vs ham

Data source: https://s3-us-west-2.amazonaws.com/ga-dat-2015-suneel/datasets/sms_spam_or_ham.csv

Description of data: each row is of the form "label, body" where label is either spam or ham (not spam) and body is the SMS body.
Description of challenge: Using code similar to what we did for Twitter sentiment analysis, create a classifier and experiment with the feature set to most accurately predict whether an SMS is spam or ham.


Let's use Naive Bayes for SMS Filtering!
Size of our data set: 5563
Now training the classifier and testing the accuracy...
The accuracy was.... 0.94967423051
It took 0.275185108185 seconds to run the algorithm

Most Informative Features
        contains_weblink = True             spam : ham    =     46.2 : 1.0
     contains_spam_words = True             spam : ham    =     26.4 : 1.0
                  length = 'medium'          ham : spam   =      9.2 : 1.0
                  length = 'long'           spam : ham    =      3.0 : 1.0
     contains_spam_words = False             ham : spam   =      2.4 : 1.0
        contains_weblink = False             ham : spam   =      1.5 : 1.0
        
        
Ipython Notebook: http://nbviewer.ipython.org/gist/anandshudda/1ef24148f79634f6e62d
