import csv
import nltk
import random
import time

from itertools import chain
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

from copy import deepcopy


########################################################
def get_tfidf_features():
    
    f = open('/home/vagrant/repos/datasets/clean_twitter_data.csv', 'rb')

    rows = []
    data = []

    for row in csv.reader(f):
        rows.append(row[1])
        data.append(row)
    
    corpus = rows[:10000]
    all_data = data[:10000]
    
    vectorizer = TfidfVectorizer(ngram_range=(1,1), min_df=1, stop_words='english', max_features=5000, analyzer='word', strip_accents='ascii')
    vectorizer.fit_transform(corpus)

    features = vectorizer.get_feature_names()
    
    feature_dict = dict((f, False) for f in features)
    
    return all_data, feature_dict
    
########################################################
    

########################################################
def twitter_features(tweet, d_f_t):
    
    u_d_f_t = deepcopy(d_f_t)
    
    for word in tweet.split():
        if word in u_d_f_t:
            u_d_f_t[word] = True

    return u_d_f_t
########################################################

########################################################
def get_feature_sets(tweets, def_feature_dict):

    output_data = []

    for t in tweets:    
        label = t[0]
        
        updated_feature_dict = twitter_features(t[1], def_feature_dict)
        
        # add the tuple of feature_dict, label to output_data
        data = (updated_feature_dict, label)
        
        output_data.append(data)
        
    return output_data
########################################################

########################################################
def get_training_and_validation_sets(feature_sets):

    random.shuffle(feature_sets)

    count = len(feature_sets)
    
    slicing_point = int(.20 * count)

    training_set = feature_sets[:slicing_point]

    validation_set = feature_sets[slicing_point:]
    
    return training_set, validation_set
########################################################

########################################################
def run_classification(training_set, validation_set):

    classifier = nltk.NaiveBayesClassifier.train(training_set)
    
    accuracy = nltk.classify.accuracy(classifier, validation_set)
    
    print "The accuracy was.... {}".format(accuracy)
    
    return classifier
########################################################

########################################################
def predict(classifier, new_tweet):
    
    return classifier.classify(twitter_features(new_tweet))
########################################################

########################################################
# Now let's use the above functions to run our program
start_time = time.time()

print "Let's use Naive Bayes!"

tweet_data, default_feature_set = get_tfidf_features()

our_feature_sets = get_feature_sets(tweet_data, default_feature_set)

our_training_set, our_validation_set = get_training_and_validation_sets(our_feature_sets)

print "Now training the classifier and testing the accuracy..."
classifier = run_classification(our_training_set, our_validation_set)

end_time = time.time()

completion_time = end_time - start_time
print "It took {} seconds to run the algorithm".format(completion_time)
########################################################
