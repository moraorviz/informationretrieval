from .logging_helper import logger
from .textrank import TextRankClient
from .rootloglikelihood import RootLogClient
import math


def _cusfloformat(floatn):
    '''Custom float numbers formatter.'''
    return '{:.3f}'.format(floatn)


class RankedBiasedOverlap:
    '''Implementation of the similarity measure algorithm applied for
    comparing ranked lists of equal length.
    '''

    def __init__(self, list_a, list_b):
        logger.debug('Initializing %s.', self.__class__.__name__)
        self.list_a = list_a
        self.list_b = list_b

    def convert(self, list, depth):
        ans = set()
        for v in list[:depth]:
            ans.add(v)

        return ans

    def overlap(self, depth):
        '''Calculates the overlap in the two lists.'''
        
        
        set1, set2 = self.convert(self.list_a, depth), self.convert(self.list_b, depth)

        return len(set1.intersection(set2)), len(set1), len(set2)

    def rbo_min(self, p, depth=None):
        '''Tight lower bound on RBO. Equation (11) of the paper.'''

        logger.debug('Executing %s.', self.rbo_min.__name__)

        depth = min(len(self.list_a), len(self.list_b)) if depth is None else depth
        x_k = self.overlap_wties(depth)
        log_term = x_k * math.log(1-p)
        sum_term = sum(
            p**d/d*(self.overlap_wties(d) - x_k) for d in range(1, depth + 1)
        )

        return (1 - p) / p * (sum_term - log_term)

    def agreement(self, depth):
        '''Proportion of shared values between lists.'''

        len_int, len_set1, len_set2 = self.overlap(depth)
        return 2*len_int/(len_set1 + len_set2)

    def overlap_wties(self, depth):
        ''' Small modification of the overlap method.'''

        return self.agreement(depth)*min(depth, len(self.list_a), len(self.list_b))



class RBOClient:
    '''A client for the RBO algorithm.'''

    def __init__(self):
        self.textrank_client = TextRankClient()
        self.rootlog_client = RootLogClient()
        self.rbo = None

    def get_list_tr(self):
        tr_list = self.textrank_client.get_words()
        return [e[0] for e in tr_list]

    def get_list_rl(self):
        rl_list = self.rootlog_client.get_ranked_words()
        return [e[0] for e in rl_list]

    def set_rbo(self):
        logger.debug('Executing %s.', self.set_rbo.__name__)
        self.rbo = RankedBiasedOverlap(
            self.get_list_rl(), self.get_list_rl())

    def get_rbo(self):
        return self.rbo
        
# TODO: get the output lists persisted somewhere. Calulating them
# everytime from scratch is not efficient due to the time we have
# to spend.
def main():
    rboclient = RBOClient()
    rboclient.set_rbo()
    rbo = rboclient.get_rbo()
    a, b, c = rbo.overlap(1000)
    print(_cusfloformat(rbo.rbo_min(.8, 15)))

