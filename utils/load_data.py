import pandas as pd

from logging import Formatter, FileHandler, getLogger

LOG_DIR = '../logs/'
TRANSACTION_DATA = '~/handm/data/input/tran_sample.feather'
CUSTOMER_DATA = '~/handm/data/input/cust_with_sex.feather'
ARTICLE_DATA = '~/handm/data/input/arti_sample.feather'
TEST_DATA = '~/handm/data/input/sample_submission.feather'


logger = getLogger(__name__)
log_fmt = Formatter(
    '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')

handler = FileHandler(LOG_DIR + 'load_data.py.log', mode='w')
handler.setLevel('DEBUG')
handler.setFormatter(log_fmt)
logger.setLevel('DEBUG')
logger.addHandler(handler)


def read_csv(path, encoding='utf-8'):
    logger.debug('enter')
    df = pd.read_csv(path, encoding=encoding)
    logger.debug('exit')
    return df


def load_transaction_data():
    logger.debug('enter')
    df = pd.read_feather(TRANSACTION_DATA)
    logger.debug('exit')
    return df


def load_customer_data():
    logger.debug('enter')
    df = pd.read_feather(CUSTOMER_DATA)
    logger.debug('exit')
    return df


def load_article_data():
    logger.debug('enter')
    df = pd.read_feather(ARTICLE_DATA)
    logger.debug('exit')
    return df


def load_submission_data():
    logger.debug('enter')
    df = pd.read_feather(TEST_DATA)
    logger.debug('exit')
    return df


if __name__ == '__main__':
    print(load_transaction_data().head())
    print(load_customer_data().head())
    print(load_article_data().head())
    print(load_submission_data().head())
