'''A module with classes and functions to retrieve and
 manipulate the data sources.'''

import itertools
import bz2
import json
import collections

from . import logging_helper as lh
from . import clean


class Fetch:
    '''A class to operate on the data source file of reddit's posts.'''

    TEXT_ID = 'selftext'
    DOMAIN_ID = 'domain'
    ID_ID = 'id'

    def __init__(self, source, topic, ranked_list=None):
        lh.logger.debug('Initializing %s.', self.__class__.__name__)
        self.source = source
        self.topic = topic
        self.ranked_list = ranked_list
        self.posts = []

    def get_posts(self, nlines=50000):
        '''Returns a list of fetched posts.'''

        lh.logger.debug('Executing %s.', self.get_posts.__name__)

        posts = []

        with bz2.open(self.source, 'rt') as reddit_file:
            for line in itertools.islice(reddit_file, 0, nlines):
                dataset = json.loads(line)
                if dataset[self.DOMAIN_ID] == self.topic:
                    id = dataset[self.ID_ID]
                    text = dataset[self.TEXT_ID]
                    custompost = CustomPost(id, text)
                    if self.ranked_list != None:
                        custompost.set_score(self.ranked_list)
                    posts.append(custompost)

        return posts


class Word:
    '''A class to extract tokens from a list of posts.
    
    Attributes
    ----------
    posts : list
        A list of posts.
    '''

    def __init__(self, posts):
        lh.logger.debug('Initializing %s.', self.__class__.__name__)
        self.posts = posts

    def get_words(self):
        '''Extracts a dict of words with occurrences from the posts.'''

        lh.logger.debug('Executing %s.', self.get_words.__name__)

        words = []
        for post in self.posts:
            for sentence in post.text:
                for word in sentence:
                    words.append(word)

        words = collections.Counter(words)

        return words

    def get_sentences(self):
        '''Returns a list of sentences.'''

        lh.logger.debug('Executing %s.', self.get_tokens.__name__)

        sentences = []
        for post in self.posts:
            for sentence in post.text:
                sentences.append(sentence)
        
        return sentences

    def get_tokens(self):
        '''Returns a list of tokens.'''

        tokens = []
        for post in self.posts:
            for sentence in post.texst:
                for word in sentence:
                    tokens.append(word)
            
        return tokens


class CustomPost:
    '''A custom model for the Reddit posts.'''

    def __init__(self, id, text):
        self.id = id
        self.text = clean.cleantext(text)
        self.score = 0

    def set_score(self, ranked_words):
        '''Calculates and assign the score to itself according to a
        provided reference of ranked words.

        Parameters
        ----------
        ranked_words : dict
            Dictionary with words and their scores.
        '''

        for sentence in self.text:
            for word in sentence:
                if word in ranked_words:
                    self.score += ranked_words[word]