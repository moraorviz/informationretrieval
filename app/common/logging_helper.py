'''Custom logging module.

Define our own logger by creating an object of the
Logger class. It's configured using Handlers and Formatters instead of
basicConfig() method.

'''

import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

c_handler = logging.StreamHandler()
c_handler.setLevel(logging.DEBUG)

c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

c_handler.setFormatter(c_format)

logger.addHandler(c_handler)
logger.debug('Custom logger is ready.')
