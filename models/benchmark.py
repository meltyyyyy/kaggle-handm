# %%
import sys, os
sys.path.append(os.pardir)

# %%
from utils.load_data import load_transaction_data
from logs.logger import get_logger
import pandas as pd

# %%
LOG_FILE = '../logs/benchmark.py.log'

# %%
logger = get_logger(module_name=__name__, log_file=LOG_FILE)
logger.info('start')

# %%
logger.info('start loading')
tran_df = load_transaction_data()
tran_df['article_id'] = tran_df['article_id'].astype('str')
tran_df['t_dat'] = pd.to_datetime(tran_df['t_dat'])
tran_df['month'] = tran_df['t_dat'].dt.month
logger.info(f'tran_df shape: {tran_df.shape}')
logger.info(f'tran_df sample: \n{tran_df.head()}')

# %%
logger.info('oldest date: {}, latest date: {}'.format(tran_df['t_dat'].min(), tran_df['t_dat'].max()))

# %%
tran_df = tran_df.loc[tran_df['month'].isin([4, 5, 6, 7, 8, 9, 10])]
tran_df = tran_df.reset_index(drop=True, inplace=True)
logger.info('only use summer period: \n{}'.format(tran_df['month'].value_counts()))


# %%
