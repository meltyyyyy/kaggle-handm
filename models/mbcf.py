import logging
from logging import getLogger, FileHandler, Formatter
from utils.load_data import load_transaction_data, load_article_data

LOG_DIR = '../logs/'

logger = getLogger(__name__)
log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s')

handler = FileHandler(LOG_DIR + 'mbcf.py.log', mode='w')
handler.setLevel(logging.DEBUG)
handler.setFormatter(log_fmt)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

logger.info('start')
tran_df = load_transaction_data()
logger.info(f'tran_df shape: {tran_df.shape}')
logger.info(f'tran_df sample: \n{tran_df.head()}')
arti_df = load_article_data()
logger.info(f'tran_df shape: {arti_df.shape}')
logger.info(f'tran_df sample: \n{arti_df.head()}')
