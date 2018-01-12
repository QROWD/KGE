import logging

logging.basicConfig()
logger = logging.getLogger("tensor")
logger.setLevel(logging.DEBUG)
logger.propagate = False

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

logger.addHandler(ch)
