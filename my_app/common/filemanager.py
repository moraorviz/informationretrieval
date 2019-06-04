# Import python core libraries.
import operator
# Import external libraries.
import nltk
# Import custom logger
from .logging_helper import logger


class FileManager:
    '''
    Utility class to operate with files and data
    structures.

    Parameters
    ----------
    source : str
    output : str

    Methods
    -------
    order_scores(scores_dict, n_results=1500)
    save_dict(my_dict, my_file)
    save_list(my_list, my_file)
    printdict(my_dict)
    download_stopwords()
    '''

    def __init__(self, source, output):
        logger.debug('Initializing %s', self.__class__.__name__)
        self.source = source
        self.output = output

    # Orders a dictionary and returns a list.
    def order_scores(self, my_dict, n_results=1500):

        sorted_scores = sorted(my_dict.items(),
                               key=operator.itemgetter(1), reverse=True)

        return sorted_scores[:n_results]

    # Saves the contents of a dictionary into a text file.
    def save_dict(self, my_dict, my_file):

        sorted_results_list = self.order_scores(my_dict)

        with open(my_file, 'w') as f:
            logger.debug('Saving results into %s.', my_file)
            f.write('\n'.join('%s %s' % x for x in sorted_results_list))

    # Variant of the save_dict() method.
    def save_dict_variant(self, my_dict, my_file):
        with open(my_file, 'w') as f:
            for word, rank in my_dict.items():
                f.write('{} {}\n'.format(word, rank))

    # Saves the contents of a list into a text file.
    def save_list(self, my_list, my_file):
        with open(my_file, 'w') as f:
            for item in my_list:
                f.write('%s\n' % item)

    # Prints the contents of a dict to the console.
    def printdict(self, my_dict):

        for key, value in dict.items():
            print(f'{key:<4} {value}')

    # Downloads a set of common words from the web. One time only execution.
    def download_stopwords():
        nltk.download('stopwords')
