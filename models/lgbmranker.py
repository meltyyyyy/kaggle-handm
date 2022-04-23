# %%
import sys
import os
sys.path.append(os.pardir)

# %%
from logs.logger import get_logger
from utils.load_data import load_transaction_data, load_article_data, load_customer_data

# %%
LOG_FILE = '../logs/lgbmranker.py.log'

# %%
logger = get_logger(module_name=__name__, log_file=LOG_FILE)
logger.info('start')


# %%
logger.info('load data')
tran_df = load_transaction_data()
arti_df = load_article_data()
cust_df = load_customer_data()
logger.info(f'tran_df shape: {tran_df.shape}')
logger.info(f'tran_df head: \n{tran_df.head()}')
logger.info(f'arti_df shape: {arti_df.shape}')
logger.info(f'arti_df head: {arti_df.head()}')
logger.info(f'cust_df shape: {cust_df.shape}')
logger.info(f'cust_df head: {cust_df.head()}')

# %%
