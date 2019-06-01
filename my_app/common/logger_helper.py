'''logging_helper.py

Define our own logger by creating an object of the
Logger class. It's configured using Handlers and Formatters instead of
basicConfig() method.
'''

import logging

# Change the root logger default mode.
# Without this line the custom logger prints nothing despite
# of the c_handler in DEBUG level we define below.
logging.basicConfig(level=logging.DEBUG)

# Create a custom logger.
logger = logging.getLogger(__name__)

# Create handlers
# c = console.
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.DEBUG)

# Create formatters and add it to handlers.
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)

# Add handlers to the logger
logger.addHandler(c_handler)
