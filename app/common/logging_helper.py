'''Custom logging module.

Define our own logger by creating an object of the
Logger class. It's configured using Handlers and Formatters instead of
basicConfig() method.

'''

import logging

# Create a custom logger and set level to debug.
# Singleton.
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler.
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.DEBUG)

# Create formatters and add it to handlers.
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)

# Add handlers to the logger
logger.addHandler(c_handler)

logger.debug('Custom logger is ready.')
