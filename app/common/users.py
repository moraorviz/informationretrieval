'''This module implements some ML methods to detect users who exhibit signs of depression.'''

import csv
import random

from . import data
from . import logging_helper as lh

import numpy as np
import sklearn as skl
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


CAT_FILE = 'app/resources/allsubreddits.csv'
CATEGORIES = ['Depression', 'NotDepression']
SOURCE_FILE = 'app/resources/RS_2017-10.bz2'
DEPTOPICS = ['add', 'cripplingalcoholism', 'disorder', 'Health',
'HealthProject', 'leaves', 'MenGetRapedToo', 'rapecounseling',
'7CupsofTea', 'addiction', 'ADHD', 'Advice', 'affirmations', 'afterthesilence',
'Agoraphobia', 'AlAnon', 'alcoholicsanonymous', 'alcoholism', 'Anger', 'Antipsychiatry',
'Anxiety', 'Anxietyhelp', 'anxietysuccess', 'anxietysupporters', 'AskDocs',
'aspergers', 'AspiePartners', 'AtheistTwelveSteppers', 'behavior', 'behaviortherapy',
'bingeeating', 'BipolarReddit', 'BipolarSOs', 'BodyAcceptance', 'BPD', 'bulimia',
'CompulsiveSkinPicking', 'dbtselfhelp', 'depression', 'depressionregimens',
'disability', 'distractit', 'domesticviolence', 'downsyndrome', 'DysmorphicDisorder',
'eating_disorders', 'EatingDisorderHope', 'EatingDisorders', 'emetophobia',
'EOOD', 'ForeverAlone', 'fuckeatingdisorders', 'GetMotivated', 'getting_over_it',
'GFD', 'HaveHope', 'HealthAnxiety', 'helpmecope', 'itgetsbetter', 'leaves',
'mentalhealth', 'mentalillness', 'mentalpod', 'mixednuts', 'MMFB', 'MSTsurvivors',
'needadvice', 'Needafriend', 'neurodiversity', 'NoFap', 'nosurf', 'OCD', 'OCPD',
'offmychest', 'OpiatesRecovery', 'PanicParty', 'Phobia', 'PsychiatricFreedom',
'Psychiatry', 'psychology', 'psychopathology', 'psychotherapy', 'psychotic_features',
'psychoticreddit', 'ptsd', 'quittingkratom', 'rape', 'rapecounseling', 'reasonstolive',
'rehabtherapy', 'sad', 'schizoaffective', 'schizophrenia', 'secondary_survivors',
'selfharm', 'SelfHarmCommunity', 'selfhelp', 'siblingsupport', 'slp', 'SMARTRecovery',
'socialanxiety', 'socialskills', 'socialwork', 'socialworkresources', 'specialed',
'StopDipping', 'stopdrinking', 'stopgaming', 'StopSelfHarm', 'stopsmoking',
'StopSelfHarm', 'stopsmoking', 'SuicideWatch', 'survivorsofabuse', 'swami', 'Teetotal',
'TheMixedNuts', 'tOCD', 'Tourettes', 'traumatoolbox', 'Trichsters', 'TwoXADHD',
'uniqueminds', 'whatsbotheringyou'] 


def process_csv(csvfile):
    '''Prints the contents of a csv file through console.'''

    with open(csvfile) as cfile:
        topicsreader = csv.reader(cfile)
        for row in topicsreader:
            print(', '.join(row))


def prepare_notdepression():
    '''Returns a list of 100 posts of the category NotDepression.'''

    # Fetch for non-depression topics.
    fetch = data.FetchMultiple(SOURCE_FILE, DEPTOPICS)
    posts = fetch.get_posts()
    outlist = random.sample(posts, 100)

    return outlist 


def prepare_depression(n):
    '''Returns a colleciton of posts of the category Depression.'''

    top = data.Top()
    posts = top.get_positives(n)

    return posts 


def prepare_input(n=500000):
    '''Calls other functions to extract collections of two families of posts: Depression and NotDepression.'''

    lh.logger.debug('Executing %s.', prepare_input.__name__)

    # TODO: use of random library to extract random posts for the big dataset.
    notdepr = prepare_notdepression()
    depr = prepare_depression(n)

    return depr, notdepr


class TrainData:
    '''A class to implement the logic neccesary to classify the datasets. Uses scikit-learn toolkit.'''

    def __init__(self, lista, listb):
        lh.logger.debug('Initializing %s.', self.__class__.__name__)
        self.lista = lista
        self.listb = listb
        self.categories = CATEGORIES

    def createBunch(self):
        lh.logger.debug('Executing %s.', self.createBunch.__name__)
        merged_lst = self.lista + self.listb
        random.shuffle(merged_lst)
        lh.logger.debug('Size of merged list: %s', len(merged_lst))
        category_codes = []

        for post in merged_lst:
           if type(post) == data.CustomPostWDepres:
               category_codes.append(2)
           else:
               category_codes.append(1)

        data_lst = []
        filenames = []
        for post in merged_lst:
            data_lst.append(post.originaltext)
            filenames.append(post.id)

        train = skl.datasets.lfw.Bunch(
            target_names = self.categories,
            target = np.array(category_codes),
            data = data_lst,
            filenames = filenames
        )

        # Test the method.
        print(train.target[:10])
        print("\n".join(train.data[0].split('\n')[:3]))
        index = train.target[0] - 1
        print(train.target_names[index])

        for t in train.target[:10]:
            index = t - 1
            print(train.target_names[t - 1])

        return train

    def train(self, train):
        '''Creates the bag of words.'''

        # Tokenizing text.
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(train.data)
        # From occurrences to frequencies.
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        # Training a classifier.
        clf = MultinomialNB().fit(X_train_tfidf, train.target)
        docs_new = ['Today I am having an interview.', 'Today I have bouth a new computer.']
        X_new_counts = count_vect.transform(docs_new)
        X_new_tfidf = tfidf_transformer.transform(X_new_counts)
        predicted = clf.predict(X_new_tfidf)

        for doc, category in zip(docs_new, predicted):
            # Offset for the index number mismatch.
            index = category -1
            print('%r => %s' % (doc, train.target_names[index]))

        # TODO: implement pipeline and do the evaluation of the performance on the test set.


def main():

    # Prepare input datasets.
    depr, notdepr = prepare_input()
    # Instance train data object.
    traindata = TrainData(depr, notdepr)
    # Call createBunch() method.
    train = traindata.createBunch()
    # Call createBagOfWords() method.
    traindata.train(train)

main()