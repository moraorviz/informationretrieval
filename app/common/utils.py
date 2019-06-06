'''A module containing utility functions to deal with data structures.'''


import operator


def dict2list(mydict):
    '''Transforms a dictionary into a list of (key, value) pairs.

    Returns a list.
    '''

    outlist = []
    for key, value in mydict.items():
        outlist.append((key, value))

    return outlist


def printdict(mydict):
    '''Prints the contents of mydict to the console with a proper string formatting.

    Returns nothing.
    '''

    for key, value in mydict.items():
        print(f'{key:<4} {value}')


def order_dict_scores(my_dict, n_results=1500):
    '''Returns an ordered list of (key, value) pairs according to the value amount
     in descending order.

    Returns the 1500 top scored items by default.
    '''

    sorted_scores = sorted(
        my_dict.items(), key=operator.itemgetter(1), reverse=True)

    return sorted_scores[:n_results]


def string_freq_toint(collection):
    '''Converts the frequency values in the collection to integer type.

    Returns the colleciton with the values type fixed.
    '''

    for word in collection.keys():
        collection[word] = int(collection[word])

    return collection


def remove_nestings(inlist, outlist):
    '''Recursive method to convert nested lists into a flat list.'''

    for item in inlist:
        if type(item) == list:
            remove_nestings(item, outlist)
        else:
            outlist.append(item)

def order_posts(posts_list):
    '''Returns an ordered lists of posts by score attribute by descending order.'''

    return sorted(posts_list, key=lambda x: x.score, revese = True)
    
def order_list(my_list):
    '''Orders a list of tuples by second item'''

    outlist = sorted(my_list, key=lambda x: x[1], reverse=True)

    return outlist
