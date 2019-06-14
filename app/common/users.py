'''This module implements some ML methods to detect users who exhibit signs of depression.'''

import csv
import random
import pdb

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

i = 0
for element in DEPTOPICS:
    DEPTOPICS[i] = 'self.' + element

TOPICS_UNDER_STUDY = ['self.addiction', 'self.alcoholism', 'self.stopgaming']

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
    '''Returns a collection of posts of the category Depression.'''

    top = data.Top()
    posts = top.get_positives(n)

    return posts 


def prepare_posts_under_study():
    '''Returns a collection of posts belonging to one of the categories under study.'''

    # Fetch the posts through our fetcher.    
    fetch_addiction = data.Fetch(SOURCE_FILE, 'self.addiction')
    # fetch_alcoholism = data.Fetch(SOURCE_FILE, 'self.alcoholism')
    # fetch_stopgaming = data.Fetch(SOURCE_FILE, 'self.stopgaming')
    # fetch_stopsmoking = data.Fetch(SOURCE_FILE, 'self.stopsmoking')
    # Choose a higher n for more results?
    posts_addiction = fetch_addiction.get_posts(500000)
    # posts_alcoholism = fetch_alcoholism.get_posts(5000000)
    # posts_stopgaming = fetch_stopgaming.get_posts(5000000)
    # posts_stopsomking = fetch_stopsmoking.get_posts(5000000)

    return posts_addiction


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

        return train

    def train(self, train, input_txt_lst):
        '''Creates the bag of words.'''

        # Tokenizing text.
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(train.data)
        # From occurrences to frequencies.
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        # Training a classifier.
        clf = MultinomialNB().fit(X_train_tfidf, train.target)
        X_new_counts = count_vect.transform(input_txt_lst)
        X_new_tfidf = tfidf_transformer.transform(X_new_counts)
        predicted = clf.predict(X_new_tfidf)

        for doc, category in zip(input_txt_lst, predicted):
            # Offset for the index number mismatch.
            index = category - 1
            print('%r => %s' % (doc, train.target_names[index]))

        # TODO: implement pipeline and do the evaluation of the performance on the test set.
        return predicted


def main():

    # Prepare input datasets.
    depr, notdepr = prepare_input()
    # Instance train data object.
    traindata = TrainData(depr, notdepr)
    # Call createBunch() method.
    train = traindata.createBunch()

    # Substitute with posts related with the topics separatedly.
    posts_addiction = prepare_posts_under_study()
    docs_addiction = []

    for post in posts_addiction:
        text = post.originaltext 
        docs_addiction.append(text)

    # Call createBagOfWords() method.
    predicted = traindata.train(train, docs_addiction)
    index = 0 
    dprusers = 0
    userlst = []
    # Avoid user repetition.
    # pdb.set_trace()
    for prediction in predicted:
        post = posts_addiction[index]
        user = post.user
        if prediction == 2:
            if user not in userlst and user != '[deleted]':
                index += 1
                dprusers += 1
                userlst.append(user)

    print(predicted)
    print(dprusers)
    print(len(posts_addiction))


main() 
