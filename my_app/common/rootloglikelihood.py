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
import nltk
import urllib.request
import tempfile
import math
# Import custom logger.
from .logging_helper import logger
from nltk.corpus import stopwords


class DataGenerator:
    '''
    A class for generating collections of words to be
    compared by LLRImpl class. The words are extracted from a
    reddit dataset in json format.

    Attributes
    ----------
    filename : str
        path to the source file.
    topic_name : str
        the name of the reddit topic field used as a key in the json.

    Methods
    -------
    getwords(nlines=50000)
        returns a dictionary of words and frequencies.
    '''

    # Static class members.
    stop_words = stopwords.words('english')

    def __init__(self, filename, topic_name):
        '''
        Parameters
        ----------
        filename : str
            source file path.
        '''
        logger.debug('Initializing %s.', self.__class__.__name__)

        self.text_id = 'selftext'
        self.domain_id = 'domain'
        self.depr_value = topic_name
        self.filename = filename
        self.depression_coll = collections.Counter()

    def getwords(self, nlines=50000):
        '''Uncompresses the dataset, parses the resultant json and extract
        words into a collection.

        Parameters
        ----------
        nlines : int
            scope, total number of lines to parse in the json.
        '''
        logger.debug('Executing %s.', self.getwords.__name__)

        with bz2.open(self.filename, 'rt') as reddit_file:
            for line in itertools.islice(reddit_file, 0, nlines):
                dataset = json.loads(line)
                if (dataset[self.domain_id] == self.depr_value):
                    text = dataset[self.text_id]
                    tokens = nltk.word_tokenize(text)
                    tokens = [word for word in tokens if word.isalpha()]
                    tokens = [word.lower() for word in tokens]
                    # Remove stop words
                    tokens = [word for word in tokens
                              if word not in DataGenerator.stop_words]
                    tokens = [word for word in tokens
                              if len(word) > 1]
                    self.depression_coll.update(tokens)

        return self.depression_coll


class CommonWord:
        '''
        A class for processing the list of most common words.
        Returns a dictionary of most common words and frequencies

        Attributes
        ----------
        url : string
            resource for most common words.
        commonwords_coll : dic
            container for the results.

        Methods
        -------
        getwords()
            loads contents into the class members.
        strfreqtoint(collection)
            changes the data type of values in a collection.
        '''

        def __init__(self):
            '''
            Parameters
             ----------
            none.
            '''
            logger.debug('Initializing %s.', self.__class__.__name__)

            self.url = 'http://norvig.com/ngrams/count_1w.txt'
            self.commonwords_coll = {}

        def getwords(self):
            '''Downloads the contents of the file specified in the url
            and loads them into memory inside a dict object's member
            class for further usage, basically for filtering common words.

            Parameters
            ----------
                none.
            '''

            logger.debug('Executing getwords method of %s.', self.__class__.
                         __name__)

            temp = tempfile.TemporaryFile(mode='w+t')

            try:
                with urllib.request.urlopen(self.url) as response:
                    logger.debug('Opening the temp file.')
                    html = response.read().decode('utf-8')
                    temp.writelines(html)
                    temp.seek(0)

                for line in temp:
                    word, count = line.split()
                    self.commonwords_coll[word] = count

            finally:
                logger.debug('Closing the temp file.')
                temp.close()

            return self.strfreqtoint(self.commonwords_coll)

        def strfreqtoint(self, collection):
            '''Converts frequency values to integers in a given collection.

            Parameters
            ----------
            collection: coll
                input collection.
            '''

            for word in collection.keys():
                collection[word] = int(collection[word])

            return collection


class RootLogLikelihoodRatio:
    '''
    A class for implementing the mentioned algorithm over the Reddit
    and Peter Norvig's datasets.

    Attributes
    ----------
    reddit_collection : coll
        Input collection with the words.
    common_collection: coll
        Collection of commond word for cleaning purposes.
    scores : dict
        Container for the final results.

    Methods
    -------
    calculate_score(a, b, c, d)
        Algorithm implementation.
    applyllr()
    '''

    # Class initializer method.
    def __init__(self, reddit_collection, common_collection):
        logger.debug('Initializing %s.', self.__class__.__name__)
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

        logger.debug('Executing %s method.', self.applyllr.__name__)

        # Calculate the scores for the all the dataset.
        for word, frequency in self.reddit_collection.items():

            a = frequency
            b = self.common_collection.get(word, 0)
            c = len(self.reddit_collection)
            d = len(self.common_collection)
            result = self.calculate_score(a, b, c, d)
            self.scores[word] = result

        return self.scores
