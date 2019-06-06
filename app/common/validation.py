'''This module contains functionality to calculate the F measure
of our method to detect evidence of depression in the reddit's posts.

'''

from collections import OrderedDict, Counter

from . import logging_helper as lh
from . import clean
from . import utils
from . import persistence
from . import data


TR_FILE = 'app/output/textrank.json'
RL_FILE = 'app/output/rootlog.json'
SOURCE_FILE = 'app/resources/RS_2017-10.bz2'


def combine(dict_textrank, dict_rootlog):
    '''Returns a list of 100 central terms from both outputs.'''

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

    scored_lst = []
    for i in inter:
        score = dict_b[i]
        pair = (i, score)
        scored_lst.append(pair)

    scored_dict = dict(utils.order_list(scored_lst[:100]))

    return scored_dict

def get_posts():
    '''Returns a lists of punctuated posts.'''

    posts = []

    rootlog_rw = persistence.load_json(TR_FILE) 
    textrank_rw = persistence.load_json(RL_FILE)

    # fetch = data.Fetch(SOURCE_FILE, 'self.offmychest')

    return rootlog_rw, textrank_rw


def divide_posts(olist):
    '''Returns two list with the most/less scored posts.

    Parameters
    ---------- 
    olist : list
        Ordered list of posts.
    '''

    return olist[:100], olist[:-101:-1]

def get_positives_negatives(lst):
    '''Given a list of total posts returns two lists of positive and negative cases.
    
    Parameters
    ----------
    lst : list
        Input list of total ranked posts.
    '''

    positives = []
    negatives = []
    for post in lst: 
        hit = False
        for sentence in post.cleantext:
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
    
    rootlog, textrank = get_posts()
    dic = combine(textrank, rootlog)
    utils.printdict(dic)
    print(len(dic))

main()