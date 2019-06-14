'''This module contains functionality to calculate the F measure
of our method to detect evidence of depression in the reddit's posts.

'''

import operator
import json
from collections import OrderedDict, Counter

from . import logging_helper as lh
from . import clean
from . import utils
from . import persistence
from . import data


TR_FILE = 'app/output/textrank.json'
RL_FILE = 'app/output/rootlog.json'
SOURCE_FILE = 'app/resources/RS_2017-10.bz2'


def combine(dict_textrank=None, dict_rootlog=None, save=False):
    '''Returns a list of 100 central terms from both outputs.'''

    if dict_textrank == None:
        dict_textrank = persistence.load_json('app/output/textrank.json')
    
    if dict_rootlog == None:
        dict_rootlog = persistence.load_json('app/output/rootlog.json')

    keys_a = set(dict_textrank.keys())
    keys_b = set(dict_rootlog.keys())
    intersection = keys_a & keys_b

    list_a = utils.order_list(utils.dict2list(dict_textrank))
    list_b = utils.order_list(utils.dict2list(dict_rootlog))

    dict_a = dict(list_a[:300])
    dict_b = dict(list_b[:300])

    set_a = dict_a.keys()
    set_b = dict_b.keys()
    inter = set_a & set_b

    outdct = {}
    for word in inter:
        score = dict_b[word]
        outdct[word] = score

    sorted_lst = sorted(outdct.items(), key=operator.itemgetter(1), reverse=True)

    if save:
        persistence.save_json(sorted_lst[:100], 'app/output/centralterms.json')

    return sorted_lst


def load_ranked_words():
    '''Returns dicts of punctuated words in json format.'''

    rootlog_rw = persistence.load_json(RL_FILE) 
    textrank_rw = persistence.load_json(TR_FILE)
    central_rw = persistence.load_json('app/output/centralterms.json')

    return rootlog_rw, textrank_rw, central_rw

def punctwords(save):
    '''Punctuates a list of words according to a reference list and
    saves into a file in json format.'''
    rl, tr = load_ranked_words()
    lst = combine(tr, rl)

    outdct = {}
    for word in lst:
        for key, val in rl.items():
            if word == key:
                outdct[word] = val

    if save:
        persistence.save_json(outdct, 'app/output/centralterms.json')

    return outdct


def top_posts(n=100000):
    '''Returns two lists with the most/less scored posts. Also persists
    in json format.

    Parameters
    ---------- 
    None.
    '''

    imwords = persistence.load_json('app/output/centralterms.json')
    omc_fetch = data.Fetch(SOURCE_FILE, 'self.offmychest')
    omc_posts = omc_fetch.get_posts(n, imwords)
    lh.logger.debug('Parsing %s lines.', n)
    tposts = utils.order_posts(omc_posts)
    top100 = tposts[:100]
    less100 = tposts[:-101:-1]

    return top100, less100


def get_positives(n):
    '''Calculates the true positives.    
    '''

    positives = []
    negatives = []
    top100, less100 = top_posts(n)
    topandless = top100 + less100
    for post in topandless: 
        hit = False
        for sentence in post.text:
            for word in sentence:
                if 'depres' in word:
                    hit = True
        if hit:
            positives.append(post)
        else:
            negatives.append(post)

    return positives, negatives


def get_true_positives(positives, top, less):
    '''Given a list of positives and top rated posts, calculates the number of true positives.

    Parameters
    ----------
    positives : list
        A list of positive posts. 
    top : list
        A list of top ranked posts.
    '''

    i = 0
    j = 0

    for post in positives:

        id_positive = post.id

        for post in top:
            if id_positive == post.id:
                i += 1
                print('True positive found.')

        for post in less:
            if id_positive == post.id:
                j += 1
                print('False positive found.')

    return i, j

def main():

    lh.logger.debug('Starting %s.', main.__name__)
    
    positives, negatives = get_positives(100000)
    print(len(positives))

    lh.logger.debug('Done.')


main()