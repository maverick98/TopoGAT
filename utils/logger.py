import logging

def get_logger():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    return logging.getLogger(__name__)