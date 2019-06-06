'''Text Cleaning Utilities.

This script contains functions and classes to parse
and clean text before processing.

This script can be imported as a module.
'''

import re
import contractions

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

from . import logging_helper as lh


stop_words = stopwords.words('english')
pattern = r'[^a-zA-z\s]'

def download_stopwords():
    '''Downloads the set of common words from nltk library.'''
    nltk.download('stopwords')

def clean(word):
    '''Keep letters only. Uses regular expressions.'''

    return re.sub(pattern, '', word)

def clean_sentence(sentence):
    '''Cleans a whole sentence.'''

    sentence = [clean(word) for word in sentence]
    return [word for word in sentence if word and word != '``']

def clean_sentences(sentences):
    '''Cleans a set of sentences.'''

    return [clean_sentence(sentence) for sentence in sentences]

def lower(sentence):
    '''Converts all the words in a sentence into lowercase.'''

    return [word.lower() for word in sentence]

def remove_stopwords(sentence):
    '''Removes common words from the sentence.'''

    words = [word for word in sentence if word not in stop_words]
    return [word for word in words if len(word) > 1]

def tokenize_words(sentences):
    '''Split a sentence into individual tokens.'''

    return [word_tokenize(sentence) for sentence in sentences]

def fix_contractions(sentences):
    '''Removes contractions.'''

    return [contractions.fix(sentence) for sentence in sentences]

def remove_stopwords_sent(sentences):
    '''Removes the stopwords for a set of sentences.'''

    return [remove_stopwords(sentence) for sentence in sentences]

def cleantext(text):
    '''Applies the previous functions to a text. Cleans the whole text and outputs
    a proper formatted string for processing.
    
    Return an array of tokenized sentences.
    '''

    sentences = sent_tokenize(text)
    clean1 = fix_contractions(sentences)
    clean2 = lower(clean1)
    clean3 = tokenize_words(clean2)
    clean4 = clean_sentences(clean3)
    clean5 = remove_stopwords_sent(clean4)

    return clean5

def purge_depres(word_list):
    '''Eliminates the words in the list which contain the substring 'depres'.'''

    solution_list = word_list
    for w in word_list:
        if 'depres' in w:
            solution_list.remove(w)

    return solution_list