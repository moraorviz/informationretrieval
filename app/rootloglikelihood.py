import logging
import collections
import itertools
import bz2
import json
import nltk
import urllib.request
import tempfile
import math
import operator

# basic logging configuration
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',
                    level=logging.DEBUG)


class DataGenerator:
    '''
    A class for generating collections of words to be
    compared by LLRImpl class. The words are extracted from a
    reddit dataset in json format.

    Attributes
    ----------
    filename : str
        path to the source file.

    Methods
    -------
    getwords(nlines=50000)
        returns a dictionary of words and frequencies.
    '''

    # Initializer method and initializer variables
    def __init__(self, filename):
        '''
        Parameters
        ----------
        filename : str
            source file path.
        '''

        logging.info('Executing __init__ method of %s', self.__class__.
                     __name__)

        # data attributes
        self.depression_coll = collections.Counter()
        self.text_id = 'selftext'
        self.domain_id = 'domain'
        self.depr_value = 'self.depression'
        self.filename = filename

    def getwords(self, nlines=50000):

        logging.info('Executing getwords method of %s', self.__class__.
                     __name__)

        with bz2.open(self.filename, 'rt') as reddit_file:
            for line in itertools.islice(reddit_file, 0, nlines):
                dataset = json.loads(line)
                if (dataset[self.domain_id] == self.depr_value):
                    text = dataset[self.text_id]
                    tokens = nltk.word_tokenize(text)
                    tokens = [word for word in tokens if word.isalpha()]
                    tokens = [word.lower() for word in tokens]
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
        commonwords_coll : dict
            container for the results

        Methods
        -------
        getwords()
            loads contents into the class members.
        '''

        # Initializer method
        def __init__(self):
            '''
            Parameters
             ----------
            none.
            '''

            logging.info('Executing __init__method of: ' + self.__class__.
                         __name__)

            self.url = 'http://norvig.com/ngrams/count_1w.txt'
            self.commonwords_coll = {}

        def getwords(self):
            '''Downloads the contents of the file specified in the url
            and loads them into memory inside a dict object's member
            class for further usage, basically filtering common words.

            Parameters
            ----------
                none.
            '''

            logging.info('Executing getwords method of %s', self.__class__.
                         __name__)

            temp = tempfile.TemporaryFile(mode='w+t')

            try:
                with urllib.request.urlopen(self.url) as response:
                    html = response.read().decode('utf-8')
                    temp.writelines(html)
                    temp.seek(0)

                for line in temp:
                    word, count = line.split()
                    self.commonwords_coll[word] = count

            finally:
                logging.info('Closing the temp file')
                temp.close()

            return self.strfreqtoint(self.commonwords_coll)

        # Converts frequency values to integers of the given collection
        def strfreqtoint(self, collection):
            '''Converts frequency values to integers in a given collenction.

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
    and Peter Norvig's datasets

    Attributes
    ----------

    Methods
    -------
    '''

    # Class __init__ method
    def __init__(self, reddit_collection, common_collection):

        logging.info('Executing __init__ method of %s', self.__class__.
                     __name__)

        self.reddit_collection = reddit_collection
        self.common_collection = common_collection
        self.scores = {}

    # Utility method for saving into a file in the same dir
    def savetofile(self, dict):

        with open('final_comparation.txt', 'w') as file:
            for key, value in dict.items():
                file.write(f'{key:<4} {value}')
                file.write('\n')

    # Utility method for printing dicts in the console
    def printdict(self, dict):

        for key, value in dict.items():
            print(f'{key:<4} {value}')

    # Algorithm implementation
    def calculate_score(self, a, b, c, d):

        logging.info('Executing impl method of %s',
                     self.__class__.__name__)

        E1 = c*(a+b)/(c+d)
        E2 = d*(a+b)/(c+d)
        result = 2*(a*math.log(a/E1 + (1 if a == 0 else 0))
                    + b*math.log(b/E2 + (1 if b == 0 else 0)))
        result = math.sqrt(result)

        if ((a/c) < (b/d)):
            result = -result

        return result

    # Apply the algorithm over all the words in the two datasets
    def applyllr(self):

        logging.info('Executing applyllr metho of %s',
                     self.__class__.__name__)

        for word, frequency in self.reddit_collection.items():

            a = frequency
            b = self.common_collection.get(word, 0)
            c = len(self.reddit_collection)
            d = len(self.common_collection)
            result = self.calculate_score(a, b, c, d)
            self.scores[word] = result

        return self.scores


# Main method.
def main():

    logging.info('Executing main method')
    datagenerator = DataGenerator('../../resources/RS_2017-10.bz2')
    reddit_dataset = datagenerator.getwords(1000000)
    commonword = CommonWord()
    common_dataset = commonword.getwords()
    rll = RootLogLikelihoodRatio(reddit_dataset, common_dataset)
    result_dict = rll.applyllr()
    result = sorted(result_dict.items(), key=operator.itemgetter(1))
    final_result = collections.OrderedDict(result)
    rll.printdict(final_result)
    rll.savetofile(final_result)


# Entry point.
main()
