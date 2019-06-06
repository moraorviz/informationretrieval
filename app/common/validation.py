'''This module contains functionality to calculate the F measure
of our method to detect evidence of depression in the reddit's posts.

'''

import json
import itertools
import bz2
import contractions
from collections import OrderedDict, Counter

from . import logging_helper as lh
from . import clean
from .textrank_client import TextRankClient
from .rootlog_client import RootLogClient


tr_json_file = 'app/output/textrank.json'
rl_json_file = 'app/output/rootlog.json'

# Persist results in json files.
# One time execution.
# TODO check if the files are already there with data as a first step.
def persist_results():
    # Create the objects
    word = Word()
    # Get results from the algorithms.
    tr_lst = word.get_trlist()
    rl_lst = word.get_rllist()
    # Save the results.
    filemanager.save_json(tr_lst, tr_json_file)
    filemanager.save_json(rl_lst, rl_json_file)

def get_common_list(list_a, list_b):
    '''Returns a list of 100 central terms from both outputs'''

    lh.logger.debug('Executing %s.', get_common_list.__name__)
    # Initialize empty lists
    words_a = []
    words_b = []
    # Iterate through the input lists. Append words only.
    for i, v in list_a:
        words_a.append(i)
    for i,v in list_b:
        words_b.append(i)
    # Get the intersection of sets.
    matching_words = set(words_a) & set(words_b)
    # Return the results.
    return matching_words

def punctrl(words, lroot):
    '''Assigns each word within a list it's punctuation according
    to logroot algorithm.

    Parameters
    ----------
        words : list
            Input list of words.
        lroot : dict 
            Ranked words.
    '''
    # Initalize the output lists.
    output_lst = []

    # Double iteration.
    # Iterate over the list of words.
    for w in words:
        for i, v in lroot.items():
            if w == i:
                pair = (w , v)
                output_lst.append(pair)
    
    return output_lst

def getrlwords():
    '''Get the list of rootlog ranked words.'''

    # Deserialize persisted json data into a json dictionary object in memory.
    rl_data_json = deser_json(rl_json_file)    
    tr_data_json = deser_json(tr_json_file)
    # Convert the resulting dictionary into a list for easier manipulation.
    rl_list = dict2list(rl_data_json)
    tr_list = dict2list(tr_data_json)
    # Get the common set of 100 first common terms.
    common_set = get_common_list(rl_list, tr_list)
    sliced_set = set(itertools.islice(common_set, 100))
    # Transform the set into a list for easier manipulation.
    sliced_lst = list(sliced_set)
    # Punctuate the words in the list.
    punctuated_lst = punctrl(sliced_lst, rl_data_json) 
    # Purge for trivial coincidences.
    # purged_lst = purge(sliced_lst)
    # Return the results
    return punctuated_lst


class Word:
    '''A class for invoking the neccesary algorithms to obtain the results.'''

    def __init__(self):
        lh.logger.debug('Initializing %s.', self.__class__.__name__)
        self.textrank = TextRankClient()
        self.rootlog = RootLogClient()

    def get_trlist(self):
        return self.textrank.get_ordered_output()

    def get_rllist(self):
        return self.rootlog.get_ordered_output()

    def persist_results(self):
        pass
    
class Classificator:
    '''
    Operates on the dataset and classifies the posts.

    Parameters
    ----------
    rwords : lst 
        List with ranked words according to rootlog algorithm.
    '''

    # Static class members.
    INPUT = 'app/resources/RS_2017-10.bz2'
    OUTPUT = 'app/output/classification.json'
    REDDIT_TOPIC = 'self.offmychest'
    DOMAIN_ID = 'domain'
    TEXT_ID = 'selftext'
    ID_ID = 'id'

    def __init__(self):
        lh.logger.debug('Initializing %s.', self.__class__.__name__)

    def get_posts(self, nlines = 500000):
        lh.logger.debug('Inspecting file...')
        # Initialize output list.
        o_lst = []
        # Start context manager.
        with bz2.open(self.INPUT, 'rt') as reddit_file:
            for line in itertools.islice(reddit_file, 0, nlines):
                # Count lines.
                # Load into a variable in memory as a json object.
                dataset = json.loads(line)
                # Check for the desired subreddit.
                if dataset[self.DOMAIN_ID] == self.REDDIT_TOPIC:
                    # Extract the id of the post.
                    id = dataset[self.ID_ID]
                    # Extract the text of the post.
                    text = dataset[self.TEXT_ID]
                    # Create a custom post object with the obtained data.
                    custompost = CustomPost(id, text)
                    # Append the content to a list to store the results.
                    o_lst.append(custompost)

        # Return total posts. 
        return o_lst


    

def main():
    # Get list of posts with associated score.
    lst = getposts_wscore()    
    # Sort the list in descending order.
    olist = order_posts(lst)
    # Return the top/less rated lists.
    top_rated, less_rated = divide_posts(olist)
    # Join both top and less rated lists.
    merged_list = top_rated + less_rated
    print(len(merged_list))
    # Get positives and negatives.
    positives, negatives = get_positives_negatives(merged_list)
    print(len(positives))
    print(len(negatives))
    i, j = get_true_positives(positives, top_rated, less_rated)
    print(i, j)
    
    # TODO: save the results in json format.

def getposts_wscore():
    '''Returns a list of 378 (hc) punctuated posts.

    Parameters
    ----------
    None.
    '''

    posts_out = []
    da = deser_json(rl_json_file)
    classificator = Classificator()
    posts_in = classificator.get_posts(500000)

    for post in posts_in:
        post.set_cleantext()
        post.calculate_score(da)
        posts_out.append(post)

    return posts_out

def order_posts(plist):
    '''Returns an ordered list by attribute.

    Parameters
    ----------
    plist : list
        Input list of posts.
    '''

    return sorted(plist, key=lambda x: x.score, reverse = True)

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


main()