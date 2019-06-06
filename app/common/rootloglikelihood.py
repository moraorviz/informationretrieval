'''
Exercise 1: Loglikelihood Ratio algorithm implementation for
the reddit datasets. Gets a list of most relevant words within the
depression posts together with a score value.

TODO: filter very common words. Import login from the
textrank module.
'''

import collections
import itertools
import bz2
import json
import urllib.request
import tempfile
import math

from . import logging_helper as lh
from . import utils
from . import persistence
from . import data


class CommonWord:
    '''
    A class to operate on the Peter Norvig's list of common words.

    Attributes
    ----------
    None.

    Methods
    -------
    getwords()
        Downloads the words list from the internet. 
    save()
        Saves the dict of words in json format.
    '''

    NORVIG_URL = 'http://norvig.com/ngrams/count_1w.txt'
    NORVIG_JSON_FILE = 'app/output/norvig.json'

    def __init__(self):
        lh.logger.debug('Initializing %s.', self.__class__.__name__)

    def getwords(self):
        '''Downloads the collection of words and stores it in a dict.

        Returns the dictionary.

        Parameters
        ----------
        None.
        '''

        lh.logger.debug('Executing %s.', self.getwords.__name__)

        words = {}
        temp = tempfile.TemporaryFile(mode='w+t')

        try:
            with urllib.request.urlopen(self.NORVIG_URL) as response:
                lh.logger.debug('Opening the temp file.')
                html = response.read().decode('utf-8')
                temp.writelines(html)
                temp.seek(0)

            for line in temp:
                word, count = line.split()
                words[word] = count

        finally:
            lh.logger.debug('Closing the temp file.')
            temp.close()

        formatted_words = utils.string_freq_toint(words)

        return formatted_words

    def save(self):
        '''Saves the collection of words in json format.

        Parameters
        ---------- 
        None.
        '''

        lh.logger.debug('Executing %s.', self.save.__name__)
        norvig_words = self.getwords()
        persistence.save_json(norvig_words, self.NORVIG_JSON_FILE)


class RootLog:
    '''
    A class for implementing the mentioned algorithm over the Reddit
    and Peter Norvig's datasets.

    Attributes
    ----------
    reddit_collection : coll
        Collection of reddit words and occurrences.
    common_collection: coll
        Collection of commond words and ocurrences.
    scores : dict
        Container for the final results.

    Methods
    -------
    calculate_score(a, b, c, d)
        Algorithm implementation.
    applyllr()
    '''

    def __init__(self, reddit_collection, common_collection):
        lh.logger.debug('Initializing %s.', self.__class__.__name__)
        self.reddit_collection = reddit_collection
        self.common_collection = common_collection
        self.scores = {}

    def calculate_score(self, a, b, c, d):
        '''Algorithm implementation.

        Parameters
        ----------
        a : int
            frequency of token of interest in dataset A.
        b : int
            frequency of token of interest in dataset B.
        c : int
            total number of observations in dataset A.
        d : int
            total number of observations in dataset B.
        '''

        E1 = c*(a+b)/(c+d)
        E2 = d*(a+b)/(c+d)
        result = 2*(a*math.log(a/E1 + (1 if a == 0 else 0))
                    + b*math.log(b/E2 + (1 if b == 0 else 0)))
        result = math.sqrt(result)

        '''if ((a/c) < (b/d)):
            result = -result
        '''

        return result

    def applyllr(self):
        '''Algorithm iteration to apply to the words in both datasets.

        Parameters
        ----------
        none.
        '''

        lh.logger.debug('Executing %s method.', self.applyllr.__name__)

        # Calculate the scores for the all the dataset.
        for word, frequency in self.reddit_collection.items():

            a = frequency
            b = self.common_collection.get(word, 0)
            c = len(self.reddit_collection)
            d = len(self.common_collection)
            result = self.calculate_score(a, b, c, d)
            self.scores[word] = result

        return self.scores


class RootLogClient:
    '''A client for the RootLog class.'''

    INPUT = 'app/resources/RS_2017-10.bz2'
    OUTPUT = 'app/output/rootlog.json'
    COMMON_WORDS = 'app/output/norvig.json'
    REDDIT_TOPIC = 'self.depression'

    def __init__(self):
        lh.logger.debug('Initializing %s.', self.__class__.__name__)
        self.fetch = data.Fetch(self.INPUT, self.REDDIT_TOPIC)

    def get_ranked_words(self):
        reddit_posts = self.fetch.get_posts()
        word = data.Word(reddit_posts)
        reddit_words = word.get_words()
        common_words = persistence.load_json(self.COMMON_WORDS)
        rootlog = RootLog(reddit_words, common_words)
        scores = rootlog.applyllr()
        order_scores = utils.order_dict_scores(scores)
        persistence.save_json(order_scores, self.OUTPUT)
        
        return order_scores 