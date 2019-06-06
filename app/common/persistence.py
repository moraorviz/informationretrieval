'''A module for persistence purposes.
'''

import json

from . import logging_helper as lh
from . import utils


def save_json(data, outfile):
    '''Saves the data in json dictionary into a file with json format.'''

    with open(outfile, 'w') as of:
        json.dump(dict(data), of)


def load_json(outfile):
    '''Loads the data in the specified .json file into an object in memory.'''

    with open(outfile, 'r') as rf:
        data = json.load(rf)
        return data


def save_list_txt(my_list, outfile):
    '''Saves the contents of a list into a text file with a proper formatting.

    Returns nothing.
    '''

    with open(outfile, 'w') as f:
        for item in my_list:
            f.write('%s\n' % item)


def save_dict_txt(my_dict, outfile):
    '''Saves the contents of a dictionary into a text file with the proper formatting.

    Returns nothing.
    '''

    sorted_results_list = utils.order_dict_scores(my_dict)

    with open(outfile, 'w') as f:
        f.write('\n'.join('%s %s' % x for x in sorted_results_list))


def save_dict_txt_variant(my_dict, outfile):
    '''A variant of the previous method. It saves the dict without sorting it first.

    Returns nothing.
    '''

    with open(outfile, 'w') as f:
        for word, rank in my_dict.items():
            f.write('{} {}\n'.format(word, rank))
