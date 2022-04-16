import pandas as pd

from logging import Formatter, FileHandler, getLogger

LOG_DIR = '../logs/'
TRANSACTION_DATA = '~/handm/data/input/transactions_train.feather'
CUSTOMER_DATA = '~/handm/data/input/customers.csv'
ARTICLE_DATA = '~/handm/data/input/articles.feather'
TEST_DATA = '~/handm/data/input/sample_submission.csv'


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
    df = pd.read_csv(CUSTOMER_DATA)
    logger.debug('exit')
    return df


def load_article_data():
    logger.debug('enter')
    df = pd.read_feather(ARTICLE_DATA)
    logger.debug('exit')
    return df


def load_test_data():
    logger.debug('enter')
    df = pd.read_csv(TEST_DATA)
    logger.debug('exit')
    return df


if __name__ == '__main__':
    print(load_transaction_data().head())
    print(load_customer_data().head())
    print(load_article_data().head())
    print(load_test_data().head())
