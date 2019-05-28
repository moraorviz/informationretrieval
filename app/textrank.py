'''
Exercise2: TextRank algorithm quick implementation

This module implements methods to analyze and parse text from a dataset
of reddit's comments. The objective is to get keywords or most important
words which can be relevant from the point of view of Natural Language
Processing.

'''

# import pdb
import os
import logging
import itertools
import json
import bz2
import nltk
import operator
import zipfile
import numpy as np
import re
import contractions
from string import punctuation
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, words
from collections import OrderedDict, Counter

# Logging configuration.
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)
logging.debug('==== Words processing with TextRank over Reddit datasets ====')


def print_dict(my_dict):
    '''Utility method to print dictionary values with a pretty output
    through the console.

    Parameters
    ----------
    my_dict : dict
        Input dictionary.
    '''

    sorted_list = sorted(my_dict.items(), key=operator.itemgetter(1))

    for key, value in sorted_list:
        print(f'{key:<4} {value}')


def symmetrize(a):
    '''Utility method to symmetrice a given square matrix.

    Parameters
    ----------
    a : array
        Numpy input matrix.
    '''

    return a + a.T - np.diag(a.diagonal())


def download_stop_words():
    '''Utility method to download a set of common words from nltk for
    filtering purposes. One time only execution.

    Parameters
    ----------
    None.
    '''

    nltk.download('stopwords')
    return stopwords.words('english')


def remove_nestings(sentences_list, output_list):
    '''Utility recursive method to conver nested lists into a normal list.

    Parameters
    ----------
    sentences_list: list
        Input list.
    output_list:list
        Empty list.
    '''

    for item in sentences_list:
        if type(item) == list:
            TextCleaner.remove_nestings(item, output_list)
        else:
            output_list.append(item)

    return output_list


def list_to_file(filename, my_list):
    '''Saves the contents of a list to a text file in the current directory.

    Parameters
    ----------
    filename : str
        Name of the output file.
    my_list : list
        Input list to be saved.
    '''

    with open(filename, 'w') as f:
        for item in my_list:
            f.write('%s\n' % item)


def dict2file(outputfile, my_dict):
    '''Writes the contents of a dictionary into a text file for
    analysis.

    Parameters
    ----------
    outputfile : str
        Output file name.
    my_dict : dict
        Input dictionary.
    '''

    with open(outputfile, 'w') as f:

        for word, rank in my_dict.items():
            f.write('{} {}\n'.format(word, rank))


class Words:
    '''
    Class to extract and organize the words in the reddit repository.

    Attributes
    ----------
    words_collection : Counter
        A dict subclass for counting words.
    text_id : str
        Json object key name.
    domain_id : str
        Json object key name.
    topic : str
        Json object key name.
    file_path : str
        Directory location of the source file.

    Methods
    -------
    get_text_only(nlines=50000)
        Returns a string containing the total amount of text.
    get_words()
        Returns a diccionary of the words and frequencies.

    '''

    def __init__(self, file_path, topic):
        '''
        Parameters
        ----------
        file_path : str
            Path to the dataset compressed file.
        topic : str
            Reddit topic of the forum thread
        '''

        # Class member attributes.
        logging.debug('Initializing %s.', self.__class__.__name__)
        self.words_collection = Counter()
        self.text_id = 'selftext'
        self.domain_id = 'domain'
        self.topic = topic
        self.file_path = file_path

    def get_text_only(self, nlines=50000):
        '''Returns a string containing the total amount of text.

        By default it parses 50000 lines of texts.
        Parameters
        ----------
        nlines : int, optional
            Number of lines to parse.
        '''

        # Empty string object as a container for the text.
        text_total = ''

        with bz2.open(self.file_path, 'rt') as reddit_file:
            for line in itertools.islice(reddit_file, 0, nlines):
                dataset = json.loads(line)
                if (dataset[self.domain_id] == self.topic):
                    text = dataset[self.text_id]
                    text_total += text

        return text_total

    def get_words(self, nlines):
        '''Returns a dictionary where they keys are the words
        in the text and the values are their frequencies.

        It updates the words_collection datamember with they key,
        value pairs obtained in the iterations over the text lines.

        Parameters
        ----------
        nlines: int, optional
            Number of lines to parse.
        '''

        with bz2.open(self.file_path, 'rt') as reddit_file:
            for line in itertools.islice(reddit_file, 0, nlines):
                dataset = json.loads(line)
                if (dataset[self.domain_id] == self.topic):
                    text = dataset[self.text_id]
                    tokens = word_tokenize(text)
                    tokens = [word for word in tokens if word.isalpha()]
                    tokens = [word.lower() for word in tokens]
                    self.words_collection.update(tokens)

            return self.words_collection


class VectorRepr:
    '''
    Class made to dive into the word vector representations and apply
    it to the problem.

    It can be neccesary to apply the textrank
    algorithm. We will use this class to find the vector for each
    word in our data according with the vector model used by TextRank.

    Attributes
    ----------
    None.

    Methods
    -------
    load_zip()
        Extracts the contents of the GloVe zip file.
    load_glove_vectors()
        Loads the glove vector model (GloVe) into memory.
    '''

    # Static class members.
    VECTOR_SIZE = 50
    EMPTY_VECTOR = np.zeros(VECTOR_SIZE)
    GLOVE_DIR = '../resources/glove/'
    GLOVE_ZIP = GLOVE_DIR + 'glove.6B.zip'
    glove_vectors_file = GLOVE_DIR + 'glove.6B.50d.txt'

    def __init__(self):
        '''
        Parameters
        ----------
        None.
        '''

        logging.debug('Initializing %s.', self.__class__.__name__)
        # Load the glove vectors at initialization time.
        self.glove_vectors = VectorRepr.load_glove_vectors()

    def load_zip(self):
        logging.debug('Loading Glove vector model.')
        zip_ref = zipfile.ZipFile(self.GLOVE_ZIP, 'r')
        zip_ref.extractall(self.GLOVE_DIR)
        zip_ref.close()

    @staticmethod
    def load_glove_vectors():
        '''Loads the contents of the glove pre-trained model into memory.

        Parameters
        ----------
        None.
        '''

        logging.debug('Loading Glove Model')
        with open(VectorRepr.glove_vectors_file,
                  'r', encoding='utf8') as glove_vector_file:

            model = {}

            for line in glove_vector_file:
                parts = line.split()
                word = parts[0]
                embedding = np.array([float(val) for val in parts[1:]])
                model[word] = embedding
                # print('Loaded {} words'.format(len(model)))

        return model


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

    def __init__(self, sentences):
        '''
        Parameteres
        -----------
        sentences: list
            A list of strings containing all the sentences.
        '''

        logging.debug('Initializing %s.', self.__class__.__name__)
        self.d = 0.85
        self.min_diff = 1e-5
        self.window_size = 4
        self.sentences = sentences
        self.steps = 10
        self.node_weight = None

    def get_vocabulary(self):
        '''Returns a dictionary containing all the words in the text.

        Parameteres
        -----------
        None.
        '''
        vocab = OrderedDict()
        i = 0
        for sentence in self.sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1

        return vocab

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
        for sentence in self.sentences:
            for i, word in enumerate(sentence):
                for j in range(i+1, i+self.window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)

        return token_pairs

    def get_keywords(self, number=50):
        ''' Returns the words ordered by importance.

        Parameters
        ----------
        number: int
            Maximum number of iterations.
        '''

        logging.debug('Executing get_keywords method'
                      ' with n = %s', number)

        # Container for the results.
        wordrank = {}

        # Prepare the results.
        node_weight = OrderedDict(
            sorted(self.node_weight.items(), key=lambda t: t[1],
                   reverse=True))

        for i, (key, value) in enumerate(node_weight.items()):

            wordrank[key] = value

            print(key + ' - ' + str(value))
            if i > number:
                break

        return wordrank

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

        g = symmetrize(g)

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

        logging.debug('Executing the iterate method.')
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


class TextCleaner:
    '''A utility class for text cleaning purposes.

    Attributes
    ----------
    text_data: str
        Input text to be cleaned.

    Methods
    -------
    clean(word):
        Removes characters from a given string.
    '''

    # Static class members.
    # Regular expression for keeping characters only.
    CLEAN_PATTERN = r'[^a-zA-z\s]'
    # Collection of common words to eliminate from the input data.
    stop_words = stopwords.words('english')
    STOP_WORDS = set(stop_words + list(punctuation))
    MIN_WORD_PROP, MAX_WORD_PROP = 0.1, 0.9

    def __init__(self, text_data):
        '''
        Parameters
        ----------
        text_data: str
            Input text to be processed.
        '''

        logging.debug('Initializing %s', self.__class__.__name__)
        self.text_data = text_data

    # Methods for text cleaning purposes.
    # Cleans the given world using regular expressions.
    @staticmethod
    def clean(word):
        return re.sub(TextCleaner.CLEAN_PATTERN, '', word)

    # CLeans a whole sentence.
    @staticmethod
    def clean_sentence(sentence):
        sentence = [TextCleaner.clean(word) for word in sentence]
        return [word for word in sentence if word and word != '``']

    # Cleans a set of sentences.
    @staticmethod
    def clean_sentences(sentences):
        return [TextCleaner.clean_sentence(sentence)
                for sentence in sentences]

    # Converts the words in the sentence into lower caps.
    @staticmethod
    def lower(sentence):
        return [word.lower() for word in sentence]

    # Another static cleaning method for deleting non-existant words within
    # a sentence.
    @staticmethod
    def word_in_dictionary(sentence):
        return [word for word in sentence if word in words.words()]

    def remove_stopwords_sent(sentences):
        return [TextCleaner.remove_stopwords(sentence)
                for sentence in sentences]

    # Removes the very common words in the sentence.
    @staticmethod
    def remove_stopwords(sentence):
        words = [word for word in sentence if word not
                 in TextCleaner.stop_words]
        result = [word for word in words if len(word) > 1]

        return result

    # Tokenizes the sentences into words.
    @staticmethod
    def tokenize_words(sentences):
        return [word_tokenize(sentence) for sentence in sentences]

    @staticmethod
    def fix_contractions(sentences):
        return [contractions.fix(sentence) for sentence in sentences]

    @staticmethod
    def compute_word_frequencies(word_sentences):
        '''Removes very common words that contribute nothing to the
        case study.

        Parameters
        ----------
        word_sentences: list
            Input set of sentences to clean.
        '''

        words = [word for sentence in word_sentences
                 for word in sentence
                 if word not in TextCleaner.STOP_WORDS]

        counter = Counter(words)
        limit = float(max(counter.values()))

        # Calculate words frequencies.
        word_frequencies = {word: freq/limit for word, freq
                            in counter.items()}

        # Drop words if too common or too uncommon.
        word_frequencies = {word: freq
                            for word, freq in word_frequencies.items()
                            if freq > TextCleaner.MIN_WORD_PROP and
                            freq < TextCleaner.MAX_WORD_PROP}

        return word_frequencies

    def process_text_sentences(self):
        '''Uses the static methods to clean the text dataset.

        Operates and updates the value of the class member text_data.

        Parameters
        ----------
        None.
        '''

        logging.debug('Processing sentences.')
        # Breakpoint.
        # pdb.set_trace()
        # Uses sent_tokenize method from nltk
        sentences = sent_tokenize(self.text_data)
        # Applies the previously defined static methods.
        clean1 = TextCleaner.fix_contractions(sentences)
        clean2 = TextCleaner.lower(clean1)
        clean3 = TextCleaner.tokenize_words(clean2)
        clean4 = TextCleaner.clean_sentences(clean3)
        clean5 = TextCleaner.remove_stopwords_sent(clean4)

        return clean5


# Main method definition.
def main():

    # Set up the directory variables.
    # TODO build a UI for allowing the use specify this
    # atm hardcored values.
    current_dir = os.getcwd()
    project_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    source_dir = project_dir + '/resources/'
    output_dir = project_dir + '/output/'
    source_file_name = 'RS_2017-10.bz2'
    output_file_name = 'textrank_output.txt'
    source_file_path = source_dir + source_file_name
    output_file_path = output_dir + output_file_name

    # Set up the topic of the target user posts.
    reddit_topic = 'self.depression'

    # Instance the words object with the source data file.
    words = Words(source_file_path, reddit_topic)

    # Get a string object containing all the texts.
    my_text = words.get_text_only()

    # Clean the texts using the text_cleaner object.
    text_cleaner = TextCleaner(my_text)
    sentences = text_cleaner.process_text_sentences()

    # Instantiate the text_rank object with the sentences.
    text_rank = TextRank(sentences)
    my_matrix = text_rank.get_matrix()

    # Performing the iteration steps.
    text_rank.iterate(my_matrix)

    # Get end results and show in the console.
    wordrank = text_rank.get_keywords(1800)
    dict2file(output_file_path, wordrank)

    logging.info('Done.')


# Main mathod call.
main()
