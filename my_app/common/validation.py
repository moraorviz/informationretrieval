# Own modules.
from .logging_helper import logger
from .textrank_client import TextRankClient
from .rootlog_client import RootLogClient
from .filemanager import FileManager
# Core libraries.
import re
import json
import itertools
import re
import bz2
import contractions
from collections import OrderedDict, Counter
# External libraries.
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Define some variables.
tr_json_file = 'my_app/output/textrank.json'
rl_json_file = 'my_app/output/rootlog.json'

# Persist results in json files.
# One time execution.
# TODO check if the files are already there with data as a first step.
def persist_results():
    # Create the objects
    word = Word()
    filemanager = FileManager()
    # Get results from the algorithms.
    tr_lst = word.get_trlist()
    rl_lst = word.get_rllist()
    # Save the results.
    filemanager.save_json(tr_lst, tr_json_file)
    filemanager.save_json(rl_lst, rl_json_file)

def get_common_list(list_a, list_b):
    '''Returns a list of 100 central terms from both outputs'''

    logger.debug('Executing %s.', get_common_list.__name__)
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

# Deserialize json
def deser_json(outfile):
    # Use the context manager to manipulate in memory the data that
    # is stored on disk.
    with open(outfile, 'r') as read_file:
        data = json.load(read_file)
        return data

# Saves data in a json objet to a file in the filesystem.
# If the file doesn't exist it should create it.
def save_json(data, outfile):
    with open(outfile, 'w+') as of:
        json.dump(dict(data), of)

# Transforms a dictionary data structure into a list for easier manipulation.
def dict2list(mydict):
    # Initialize output list.
    output_list = []
    # Iterate over the dictionary.
    for key, value in mydict.items():
        output_list.append((key, value))

    return output_list

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

def clean(pattern, word):
    '''Keep letters only. Uses regular expressions.'''

    return re.sub(pattern, '', word)

def clean_sentence(pattern, sentence):
    '''Cleans a whole sentence.'''

    sentence = [clean(pattern, word) for word in sentence]
    return [word for word in sentence if word and word != '``']

def clean_sentences(pattern, sentences):
    '''Cleans a set of sentences.'''

    return [clean_sentence(pattern, sentence) for sentence in sentences]

def lower(sentence):
    '''Converts all the words in a sentence into lowercase.'''

    return [word.lower() for word in sentence]

def remove_stopwords(sentence, stop_words):
    '''Removes common words from the sentence.'''

    words = [word for word in sentence if word not in stop_words]
    return [word for word in words if len(word) > 1]

def tokenize_words(sentences):
    '''Split a sentence into individual tokens.'''

    return [word_tokenize(sentence) for sentence in sentences]

def fix_contractions(sentences):
    '''Removes contractions.'''

    return [contractions.fix(sentence) for sentence in sentences]

def remove_stopwords_sent(sentences, stop_words):
    '''Removes the stopwords for a set of sentences.'''

    return [remove_stopwords(sentence, stop_words) for sentence in sentences]

# Eliminates words wich contain the substring 'depres'
def purge(wordlst):
    # Declare the solution list.
    sollst = wordlst
    # Iterate through the words list.
    for w in wordlst:
        # Check for the 'depres' substring.
        if 'depres' in w:
            # If match eliminate that word from the list.
            sollst.remove(w)

    return sollst 

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
        logger.debug('Initializing %s.', self.__class__.__name__)
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
    INPUT = 'my_app/resources/RS_2017-10.bz2'
    OUTPUT = 'my_app/output/classification.json'
    REDDIT_TOPIC = 'self.offmychest'
    DOMAIN_ID = 'domain'
    TEXT_ID = 'selftext'
    ID_ID = 'id'

    def __init__(self):
        logger.debug('Initializing %s.', self.__class__.__name__)

    def get_posts(self, nlines = 500000):
        logger.debug('Inspecting file...')
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

class CustomPost:
    '''A custom class to operate with posts.'''

    # Static class members
    stop_words = stopwords.words('english')
    pattern = r'[^a-zA-z\s]'

    def __init__(self, id, text):
        self.id = id
        self.text = text
        # To be calculated later in the execution.
        self.cleantext = None
        self.score = 0

    def set_cleantext(self):
        sentences = sent_tokenize(self.text)
        clean1 = fix_contractions(sentences)
        clean2 = lower(clean1)
        clean3 = tokenize_words(clean2)
        clean4 = clean_sentences(self.pattern, clean3)
        clean5 = remove_stopwords_sent(clean4, self.stop_words)

        self.cleantext = clean5

    # TODO: finish.
    def calculate_score(self, ranked_words):
        '''Assign a score to this post according to a provided reference.

        Parameters
        ----------
        ranked_words : dict
        '''

        for sentence in self.cleantext:
            for word in sentence:
                # Check for matching words.
                if word in ranked_words:
                    # Get word's punctuation and add to the post object.
                    self.score += ranked_words[word]
    

def main():

    lst = getposts_wscore()    
    olist = order_posts(lst)
    for i in olist:
        print(i.id, i.score)
    print(len(olist))

def getposts_wscore():
    '''Returns a list of 378 (hc) punctuated posts.

    Parameters
    ----------
    None.
    '''

    posts_out = []
    da = deser_json(rl_json_file)
    classificator = Classificator()
    posts_in = classificator.get_posts(10000)

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

main()