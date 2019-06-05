# import our modules
from .textrank_client import TextRankClient
from .rootlog_client import RootLogClient
from .logging_helper import logger
import scipy as sp


class Spearman:

    def __init__(self):
        logger.debug('Initializing %s.', self.__class__.__name__)
        self.textrank_client = TextRankClient()
        self.rootlog_client = RootLogClient()
        self.textrank_output = None
        self.rootlog_output = None

    def set_outputs(self):
        self.textrank_output = self.textrank_client.get_ordered_output()
        self.rootlog_output = self.rootlog_client.get_ordered_output()

    # Obtain two similar data structures from both methods in oder
    # to compare them and find the words that are present of both.
    def compare_outputs(self):

        # Iterate through the lists. Check for words that appear
        # in both lists.
        words_t = []
        words_r = []
        # Gets only they keys (the words) from the output lists.
        for i, v in self.textrank_output:
            words_t.append(i)
        for i, v in self.rootlog_output:
            words_r.append(i)

        matching_words = set(words_t) & set(words_r)

        return matching_words

    # Construct two separated lists of the matching words together
    # with it's punctuation.
    # TODO: refator repeated code.
    def build_reduced_lists(self, words_set):

        list_tr = []
        list_rl = []

        for word, rank in self.textrank_output:
            if word in words_set:
                w_tuple = (word, rank)
                list_tr.append(w_tuple)

        for word, rank in self.rootlog_output:
            if word in words_set:
                w_tuple = (word, rank)
                list_rl.append(w_tuple)

        return list_tr, list_rl

    def spy(self, my_var):
        print(type(my_var))
        print(len(my_var))

    def calculate(self, a, b):
        rho, p_value = sp.stats.spearmanr(a, b)
        return rho, p_value

    # Builds a 2D vector with the scores from both methods.
    def build_input(self, list_tr, list_rl):
        # Initialize variables.
        words = []
        tr_ranks = []
        lr_ranks = []

        for i, v in list_tr:
            words.append(i)
            tr_ranks.append(v)

        for i, v in list_rl:
            for w in words:
                if i == w:
                    lr_ranks.append(v)

        return tr_ranks, lr_ranks

    def get_values(self):
        self.set_outputs()
        matching_words_set = self.compare_outputs()
        list_tr, list_rl = self.build_reduced_lists(matching_words_set)
        x, y = self.build_input(list_tr, list_rl)
        return x, y
