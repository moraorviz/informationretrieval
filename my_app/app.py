'''app.py

Entry point of the application. Front-end logic. Console UI.
Calls the clients depending on the user input.
'''

from .common.logging_helper import logger
from .common.rootlog_client import RootLogClient
import argparse


def run():
    logger.debug('Starting the program.')
    # Instantiating the argument parser.
    parser = argparse.ArgumentParser()
    # Introducing optional arguments.
    parser.add_argument('--rootlog', help='apply rootloglikelihood '
                        'algorithm', action='store_true')
    # parser.add_argument('--textrank', help
    parser.add_argument('--textrank', help='apply textrank '
                        'algorithm', action='store_true')
    args = parser.parse_args()

    if args.rootlog:
        logger.debug('Rootlog option chosed.')
        rootlog_cli = RootLogClient()
        rootlog_cli.execute()

    if args.textrank:
        logger.debug('Textrank option chosed.')
