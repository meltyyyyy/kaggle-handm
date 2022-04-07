from logging import getLogger, Formatter, FileHandler
import logging
import pandas as pd

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
rate_df = pd.read_csv(RATE_DATA, sep='\t', encoding='latin-1', usecols=['user_id', 'movie_id', 'rating'])
user_df = pd.read_csv(USER_DATA, sep='\t', encoding='latin-1', usecols=['user_id', 'gender', 'zipcode', 'age_desc', 'occ_desc'])
movie_df = pd.read_csv(MOVIE_DATA, sep='\t', encoding='latin-1', usecols=['movie_id', 'title', 'genres'])
logger.info('rate_df shape: {}'.format(rate_df.shape))
logger.info('rate_df data samlple: \n{}'.format(rate_df.head()))
logger.info('user_df shape: {}'.format(user_df.shape))
logger.info('user_df data samlple: \n{}'.format(user_df.head()))
logger.info('movie_df shape: {}'.format(movie_df.shape))
logger.info('movie_df data samlple: \n{}'.format(movie_df.head()))
