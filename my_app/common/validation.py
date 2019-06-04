# Own modules.
from .logging_helper import logger
from .textrank_client import TextRankClient
from .rootlog_client import RootLogClient
from .filemanager import FileManager
# Core libraries.
import json

# Define some variables.
tr_json_file = 'my_app/output/textrank.json'
rl_json_file = 'my_app/output/rootlog.json'

# Persist results in json files.
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

# TODO: finish method.
def get_list(list_a, list_b):
    '''Returns a list of 100 central terms from both outputs'''

    logger.debug('Execution %s.', get_list.__name__)

# Deserialize json
def deser_json(outfile):
    # Use the context manager to manipulate in memory the data that
    # is stored on disk.
    with open(outfile, 'r') as read_file:
        data = json.load(read_file)
        return data

# Saves data in a json objet to a file in the filesystem.
# If the file doesn't exist it should create it.
# TODO: complete the method.
def save_json(data, outfile):
    with open(outfile, 'w+') as of:
        json.dump(dict(data), of)

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

def main():
    
    rl_data = deser_json(rl_json_file)    
    tr_data = deser_json(tr_json_file)
    print(tr_data)

main()