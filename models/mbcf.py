from utils.load_data import read_csv
from logging import getLogger, Formatter, FileHandler
import logging

LOG_DIR = '../logs/'
RATE_DATA = '../data/input/ml-1m/ratings.csv'
USER_DATA = '../data/input/ml-1m/users.csv'
MOVIE_DATA = '../data/input/ml-1m/movies.csv'

logger = getLogger(__name__)
log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s')

handler = FileHandler(LOG_DIR + 'mbcf.py.log', mode='w')
handler.setLevel(logging.DEBUG)
handler.setFormatter(log_fmt)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

logger.info('start')
rate_df = read_csv(RATE_DATA)
user_df = read_csv(USER_DATA)
movie_df = read_csv(MOVIE_DATA)
logger.info('rate_df shape: {}'.format(rate_df.shape))
logger.info('rate_df data samlple: \n{}'.format(rate_df.head()))
logger.info('user_df shape: {}'.format(user_df.shape))
logger.info('user_df data samlple: \n{}'.format(user_df.head()))
logger.info('movie_df shape: {}'.format(movie_df.shape))
logger.info('movie_df data samlple: \n{}'.format(movie_df.head()))
