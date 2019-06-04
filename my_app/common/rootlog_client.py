from .filemanager import FileManager
from .logging_helper import logger
from .rootloglikelihood import (DataGenerator, CommonWord,
                                RootLogLikelihoodRatio)


class RootLogClient:
    '''A class for applying some OOP principles and good practices. We
    build a client for using the logic in the RootLog module. We will
    call this client from the front-end part of the program.

    Here we can discuss about design patterns, when and how to apply them
    and how they fit to our current problem solution.
    '''

    # Static class members.
    INPUT = 'my_app/resources/RS_2017-10.bz2'
    OUTPUT = 'my_app/output/loglikelyhood.txt'
    REDDIT_TOPIC = 'self.depression'

    # Initializer method.
    def __init__(self):
        logger.debug('Initializing %s.', self.__class__.__name__)
        self.file_manager = FileManager(self.INPUT, self.OUTPUT)
        self.data_generator = DataGenerator(self.INPUT, self.REDDIT_TOPIC)
        self.common_word = CommonWord()

    # Execute the rootloglikelihoodratio logic.
    def execute(self):
        logger.debug('Executing %s method.', self.execute.__name__)
        results = self.get_output()
        self.file_manager.save_dict(results, self.OUTPUT)
        logger.debug('Finished.')

    def get_output(self):
        '''Getter method for the rootloglikelihood results.'''

        reddit_dataset = self.data_generator.getwords()
        common_dataset = self.common_word.getwords()
        root_log = RootLogLikelihoodRatio(reddit_dataset, common_dataset)
        scores_dict = root_log.applyllr()

        return scores_dict

    def get_ordered_output(self):
        scores = self.get_output()
        output = self.file_manager.order_scores(scores)

        return output
