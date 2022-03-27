import os, sys
sys.path.append(os.pardir)
import numpy as np
import pandas as pd
from tqdm import tqdm
from logging import Formatter, FileHandler, getLogger
from utils.load_data import load_transaction_data, load_test_data
tqdm.pandas()

LOG_DIR = '../logs/'

logger = getLogger(__name__)
log_fmt = Formatter(
    '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s')

handler = FileHandler(LOG_DIR + 'weekly_trend.py.log', mode='w')
handler.setLevel('DEBUG')
handler.setFormatter(log_fmt)
logger.setLevel('DEBUG')
logger.addHandler(handler)

logger.info('start')

logger.info('start loading')
transaction_df = load_transaction_data()
logger.info('finish loading')
logger.info('data shape: {}'.format(transaction_df.shape))
logger.info('data sample: \n{}'.format(transaction_df.head()))

logger.info('arrange data')

transaction_df = transaction_df.loc[:,['t_dat','customer_id','article_id']]
transaction_df['article_id'] = transaction_df['article_id'].astype(str)
transaction_df['t_dat'] = pd.to_datetime(transaction_df['t_dat'])
last_ts = transaction_df['t_dat'].max()

transaction_df['ldbw'] = transaction_df['t_dat'].progress_apply(lambda d: last_ts - (last_ts - d).floor('7D'))

weekly_sales = transaction_df.drop('customer_id', axis=1).groupby(['ldbw', 'article_id']).count()
weekly_sales = weekly_sales.rename(columns={'t_dat': 'count'})
transaction_df = transaction_df.join(weekly_sales, on=['ldbw', 'article_id'])

weekly_sales = weekly_sales.reset_index().set_index('article_id')
last_day = last_ts.strftime('%Y-%m-%d')
transaction_df = transaction_df.join(
    weekly_sales.loc[weekly_sales['ldbw']==last_day, ['count']],
    on='article_id', rsuffix="_targ")

transaction_df['count_targ'].fillna(0, inplace=True)
del weekly_sales

transaction_df['quotient'] = transaction_df['count_targ'] / transaction_df['count']

target_sales = transaction_df.drop('customer_id', axis=1).groupby('article_id')['quotient'].sum()
general_pred = target_sales.nlargest(12).index.tolist()
del target_sales

logger.info('arranged data: \n{}'.format(transaction_df))

logger.info('Fill purchase dictionary')
purchase_dict = {}

for i in tqdm(transaction_df.index):
    cust_id = transaction_df.at[i, 'customer_id']
    art_id = transaction_df.at[i, 'article_id']
    t_dat = transaction_df.at[i, 't_dat']

    if cust_id not in purchase_dict:
        purchase_dict[cust_id] = {}

    if art_id not in purchase_dict[cust_id]:
        purchase_dict[cust_id][art_id] = 0

    x = max(1, (last_ts - t_dat).days)
    a, b, c, d = 2.5e4, 1.5e5, 2e-1, 1e3
    y = a / np.sqrt(x) + b * np.exp(-c*x) - d

    value = transaction_df.at[i, 'quotient'] * max(0, y)
    purchase_dict[cust_id][art_id] += value

sub = load_test_data()

pred_list = []
for cust_id in tqdm(sub['customer_id']):
    if cust_id in purchase_dict:
        series = pd.Series(purchase_dict[cust_id])
        series = series[series>0]
        l = series.nlargest(12).index.tolist()
        if len(l) < 12:
            l = l + general_pred[:(12-len(l))]
        else:
            l = general_pred
        pred_list.append(' '.join(l))

sub['prediction'] = pred_list
sub.to_csv('submission.csv', index=None)

