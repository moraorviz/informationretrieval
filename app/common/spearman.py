'''A module for calculating the sperman coefficient of the used methods.'''


import scipy as sp
from scipy import stats

import matplotlib.pyplot as plt

from . import logging_helper as lh
from . import persistence
from . import utils


class Spearman:

    # Hard-coded data. Static class members/attributes.
    TRWORDS_FILE = 'app/output/textrank.json'
    RLWORDS_FILE = 'app/output/rootlog.json'

    def __init__(self):
        lh.logger.debug('Initializing %s.', self.__class__.__name__)
        self.trwords = persistence.load_json(self.TRWORDS_FILE)
        self.rlwords = persistence.load_json(self.RLWORDS_FILE)

    def get_common(self):

        matchwords = set(self.trwords.keys()) & set(self.rlwords.keys())
        textrank = []
        rootlog = []

        for word, rank in self.trwords.items():
            if word in matchwords:
                tpl = (word, rank)
                textrank.append(tpl)

        for word, rank in self.rlwords.items():
            if word in matchwords:
                tpl = (word, rank)
                rootlog.append(tpl)

        return textrank, rootlog

    def calculate(self, a, b):
        rho, p_value = sp.stats.spearmanr(a, b)
        return rho, p_value

    # Builds a 2D vector with the scores from both methods.
    def build_input(self, list_tr, list_rl):

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
        tr, rl = self.get_common()
        x, y = self.build_input(tr, rl)

        return x, y


class SpearmanClient:

    SCATTER_PICTURE = 'app/images/spearman.png'

    def __init__(self):
        self.spearman = Spearman()

    def get_plot(self, save=False):
        x, y = self.spearman.get_values()
        rho, p_value = self.spearman.calculate(x, y)
        print(rho)
        fig = plt.figure()
        fig.suptitle('Coeficiente de Spearman', fontsize=14, fontweight='bold')
        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.85)
        ax.set_xlabel('Puntuaciones textrank')
        ax.set_ylabel('Puntuaciones rootlog')
        scatter = plt.scatter(x, y)

        if save:
          plt.savefig(self.SCATTER_PICTURE)
        else:
          plt.show()


def main():
    spcli = SpearmanClient()
    spcli.get_plot(True)


