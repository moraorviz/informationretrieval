# Import custom logger.
from .logging_helper import logger
# Import custom modules and components.
from .filemanager import FileManager
from .textrank import Words, TextCleaner, TextRank


class TextRankClient:
    '''A client object for the TextRank logic and calculations.'''

    # Static class members.
    INPUT = 'app/resources/RS_2017-10.bz2'
    OUTPUT = 'app/output/textrank.txt'
    REDDIT_TOPIC = 'self.depression'

    def __init__(self):
        logger.debug('Initializing %s.', self.__class__.__name__)
        self.file_manager = FileManager(self.INPUT, self.OUTPUT)
        self.words = Words(self.INPUT, self.REDDIT_TOPIC)

    def execute(self):
        logger.debug('Executing %s method.', self.execute__name__)
        keywords = self.get_output()
        self.file_manager.save_dict(keywords, self.OUTPUT)
        logger.debug('Finished.')

    def get_output(self):
        text = self.words.get_text_only()
        text_cleaner = TextCleaner(text)
        sentences = text_cleaner.process_text_sentences()
        text_rank = TextRank(sentences)
        matrix = text_rank.get_matrix()
        text_rank.iterate(matrix)
        ranked_keywords_dict = text_rank.get_keywords()
        return ranked_keywords_dict

    def get_ordered_output(self):
        keywords = self.get_output()
        output = self.file_manager.order_scores(keywords)
        return output
