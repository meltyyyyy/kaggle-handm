# %%
import sys
import os
from time import time

from regex import F
sys.path.append(os.pardir)

# %%
from logs.logger import get_logger
from utils.load_data import load_transaction_data, load_article_data, load_customer_data
import pandas as pd

# %%
DRY_RUN = True
LOG_FILE = '../logs/basic_lgbmranker.py.log'

# %%
logger = get_logger(module_name=__name__, log_file=LOG_FILE)
logger.info('start')

# %%
logger.info('load data')
tran_df = load_transaction_data()
tran_df['article_id'] = tran_df['article_id'].astype('str')
tran_df['t_dat'] = pd.to_datetime(tran_df['t_dat'])
tran_df['week'] = tran_df['t_dat'].dt.week
logger.info(f'tran_df shape: {tran_df.shape}')
logger.info(f'tran_df sample: \n{tran_df.head()}')

# %%
arti_df = load_article_data()
arti_df['article_id'] = arti_df['article_id'].astype('str')
logger.info(f'arti_df shape: {arti_df.shape}')
logger.info(f'arti_df sample: \n{arti_df.head()}')

# %%
cust_df = load_customer_data()
logger.info(f'cust_df shape: {cust_df.shape}')
logger.info(f'cust_df sample: \n{cust_df.head()}')

# %%
test_week = tran_df.week.max()
logger.info(f'test week: {test_week}')
tran_df = tran_df[tran_df.week > test_week - 10]
logger.info(f'new tran_df: {tran_df.shape}')
logger.info(f'new tran_df: \n{tran_df.head()}')

# %%
cust2weeks = tran_df.groupby('customer_id')['week'].unique()
logger.info(f'c2week shape: {cust2weeks.shape}')
logger.info(f'c2week: \n{cust2weeks.head()}')

# %%
cust2weeks_shifted_weeks = {}

for cust_id, weeks in cust2weeks.items():
    cust2weeks_shifted_weeks[cust_id] = {}
    for i in range(weeks.shape[0] - 1):
        cust2weeks_shifted_weeks[cust_id][weeks[i]] = weeks[i + 1]
    cust2weeks_shifted_weeks[cust_id][weeks[-1]] = test_week

# %%
candidates_last_purchase = tran_df.copy()

weeks = []
for i, (cust_id, week) in enumerate(zip(tran_df['customer_id'], tran_df['week'])):
    weeks.append(cust2weeks_shifted_weeks[cust_id][week])

candidates_last_purchase.week = weeks

# %%
mean_price = tran_df.groupby(['week', 'article_id'])['price'].mean()
sales = tran_df.groupby('week')['article_id'].value_counts().groupby('week').rank(method='dense', ascending=False).groupby('week').head(12).rename('bestseller_rank')
bestseller_previous_week = pd.merge(sales, mean_price, on=['week', 'article_id']).reset_index()
bestseller_previous_week.week += 1
logger.info(f'bestseller_previous_week shape: {bestseller_previous_week}')
logger.info(f'bestseller_previous_week sample: \n{bestseller_previous_week.head()}')

# %%
unique_transactions = tran_df.groupby(['week', 'customer_id']).head(1).drop(columns=['article_id', 'price']).copy()
candidates_bestsellers = pd.merge(unique_transactions, bestseller_previous_week, on='week')
test_set_transactions = unique_transactions.drop_duplicates('customer_id').reset_index(drop=True)
test_set_transactions.week = test_week
candidates_bestsellers_test_week = pd.merge(test_set_transactions, bestseller_previous_week, on='week')
candidates_bestsellers = pd.concat([candidates_bestsellers, candidates_bestsellers_test_week])
candidates_bestsellers.drop(columns='bestseller_rank', inplace=True)

# %%
tran_df['purchased'] = 1
data = pd.concat([tran_df, candidates_last_purchase, candidates_bestsellers])
data.purchased.fillna(0, inplace=True)
logger.info(f'data shape: {data.shape}')
logger.info(f'data sample: \n{data.head()}')

# %%
data = pd.merge(data, bestseller_previous_week[['week', 'article_id', 'bestseller_rank']], on=['week', 'article_id'], how='left')
data = data[data.week != data.week.min()].copy()
data.bestseller_rank.fillna(999, inplace=True)
