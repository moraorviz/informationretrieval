'''
Exercise2: TextRank algorithm quick implementation

This module implements methods to analyze and parse text from a dataset
of reddit's comments. The objective is to get keywords or most important
words which can be relevant from the point of view of Natural Language
Processing.

'''

import collections
# import pdb

import numpy as np

from . import logging_helper as lh
from . import data
from . import persistence
from . import utils


class TextRank:
    '''A class for the popular algorithm implementation based on PageRank.

    Example taken from BrambleXu/TextRank4KeyWord.py in github.

    Attributes
    ----------
    d : float
        damping coefficient, usually is .85.
    min_diff : float
        convergence threshold.
    sentences : list
        a list with the sentences of the text dataset of study.
    steps : int
        number of iteration steps.
    window_size : int
        word window size.
    node_weight : None
        save keywords and it's weight

    Methods
    -------
    get_vocabulary()
        Returns an ordered dictionary of words.
    get_token_pairs():
        Returns all the generated word token pairs.
    get_matrix():
        Returns the matrix representation of the words.
    get_keywords():
        Print top number keywords.
    iterate():
        Performs the iterative steps.
    '''

    def __init__(self, word):
        '''
        Parameteres
        -----------
        sentences: list
            A list of strings containing all the sentences.
        '''

        lh.logger.debug('Initializing %s.', self.__class__.__name__)

        self.d = 0.85
        self.min_diff = 1e-5
        self.window_size = 4
        self.steps = 10
        self.node_weight = None
        self.word = word

    def symmetrize(self, a):
        '''Utility method to symmetrice a given square matrix.

        Parameters
        ----------
        a : array
            Numpy input matrix.
        '''

        return a + a.T - np.diag(a.diagonal())

    def get_token_pairs(self):
        '''Returns a list with all the tokens pairs formed from the
        set of all sentences in the text.

        According to the model there is an undirected edge between any
        two words pair. This is a quick implementation.

        Parameters
        ----------
        None.
        '''

        token_pairs = list()
        sentences = self.word.get_sentences()

        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i+1, i+self.window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)

        return token_pairs

    def get_keywords(self, total_words=1500):
        ''' Returns the words ordered by importance.

        Parameters
        ----------
        iter_number: int
            Maximum number of iterations.
        '''

        lh.logger.debug('Executing %s.', self.get_keywords.__name__)

        # Inialize empty dictionary for storing the results.
        wordrank = {}

        # Prepare the results.
        node_weight = collections.OrderedDict(
            sorted(self.node_weight.items(), key=lambda t: t[1],
                   reverse=True))

        for i, (key, value) in enumerate(node_weight.items()):

            wordrank[key] = value

            # print(key + ' - ' + str(value))
            if i >= (total_words - 1):
                break

        return wordrank


    def get_vocabulary(self):
        '''Returns a dictionary containing all the words in the text.

        Parameteres
        -----------
        None.
        '''

        vocab = collections.OrderedDict()
        sentences = self.word.get_sentences()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1

        return vocab

    def get_matrix(self):
        '''Constructs the initial transition matrix required by the model.

        Refer to the TextRank paper for full details. It uses numpy module
        for matrix operations. Returns the initial matrix.

        Parameters
        ----------
        None.
        '''

        vocab = self.get_vocabulary()
        token_pairs = self.get_token_pairs()
        vocab_size = len(vocab)

        g = np.zeros((vocab_size, vocab_size), dtype='float')

        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1

        g = self.symmetrize(g)

        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=norm != 0)

        return g_norm

    def iterate(self, g_matrix):
        '''Iterates to solve the TextRank equation using the power method.

        Updates the value of the node_weight class member.

        Parameters
        ----------
        g_matrix: array
            Initial matrix given by get_matrix method.
        '''

        lh.logger.debug('Executing %s method.', self.iterate.__name__)

        vocab = self.get_vocabulary()
        pr = np.array([1] * len(vocab))
        previous_pr = 0

        for epoch in range(self.steps):
            pr = (1-self.d) + self.d*np.dot(g_matrix, pr)
            if abs(previous_pr - sum(pr)) < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()

        for word, index in vocab.items():
            node_weight[word] = pr[index]

        self.node_weight = node_weight

    def calc(self):
        '''Perform the calculations. Returns a ranked list of words.'''

        matrix = self.get_matrix()
        self.iterate(matrix)
        keywords = self.get_keywords()

        return keywords


class TextRankClient:
    '''A client for the textrank object.'''

    INPUT = 'app/resources/RS_2017-10.bz2'
    OUTPUT = 'app/output/textrank.json'
    REDDIT_TOPIC = 'self.depression'

    def __init__(self):
        lh.logger.debug('Initializing %s.', self.__class__.__name__)
        self.fetch = data.Fetch(self.INPUT, self.REDDIT_TOPIC)

    def get_words(self):
        reddit_posts = self.fetch.get_posts()
        word = data.Word(reddit_posts)
        textrank = TextRank(word)
        scores = textrank.calc()
        order_scores = utils.order_dict_scores(scores)
        persistence.save_json(order_scores, self.OUTPUT)
        return order_scores

def main():
    trcli = TextRankClient()
    words = trcli.get_words()
    lh.logger.debug('Done.')
