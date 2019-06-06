'''app.py

Entry point of the application. Front-end logic. Console UI.
Calls the clients depending on the user input.
'''

import argparse

from .common.logging_helper import logger
from .common.rootlog_client import RootLogClient
from .common.textrank_client import TextRankClient


# TODO: add an argument to execute the spearman coefficent
# calculation and representation.
def run():
    logger.debug('Starting the program.')
    # Instantiating the argument parser.
    parser = argparse.ArgumentParser()
    # Introducing optional arguments.
    parser.add_argument('--rootlog', help='Apply rootloglikelihood '
                        'algorithm.', action='store_true')
    # parser.add_argument('--textrank', help
    parser.add_argument('--textrank', help='Apply textrank '
                        'algorithm.', action='store_true')
    args = parser.parse_args()

    if args.rootlog:
        logger.debug('Rootlog option chosed.')
        rootlog_cli = RootLogClient()
        rootlog_cli.execute()

    if args.textrank:
        logger.debug('Textrank option chosed.')
        textrank_client = TextRankClient()
        textrank_client.execute()
