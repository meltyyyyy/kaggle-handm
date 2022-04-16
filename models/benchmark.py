# %%
import sys, os
sys.path.append(os.pardir)

# %%
from utils.load_data import load_transaction_data, load_submission_data
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

# %%
tran_df['month'] = tran_df['t_dat'].dt.month
logger.info(f'tran_df shape: {tran_df.shape}')
logger.info(f'tran_df sample: \n{tran_df.head()}')

# %%
logger.info('oldest date: {}, latest date: {}'.format(tran_df['t_dat'].min(), tran_df['t_dat'].max()))

# %%
tran_df = tran_df.loc[tran_df['month'].isin([4, 5, 6, 7, 8, 9, 10])]
tran_df.reset_index(drop=True, inplace=True)
logger.info('tran_df: \n{tran_df}')

# %%
month = tran_df['month'].value_counts()
logger.info(f'number of transactions per month: \n{month}')


# %%
purchase_dict = {}

for i, t in enumerate(zip(tran_df['customer_id'], tran_df['article_id'])):
    cust_id, arti_id = t
    if cust_id not in purchase_dict:
        purchase_dict[cust_id] = {}

    if arti_id not in purchase_dict[cust_id]:
        purchase_dict[cust_id][arti_id] = 0

    purchase_dict[cust_id][arti_id] += 1


# %%
sub_df = load_submission_data()
logger.info(f'sub_df shape: {sub_df.shape}')
logger.info(f'sub_df sample: \n{sub_df.head()}')

# %%
benchmark = sub_df[['customer_id']]
pred_list = []
del list
dummy_list = list((tran_df['article_id'].value_counts()).index)[:12]
dummy_pred = ' '.join(dummy_list)

# %%
for i, cust_id in enumerate(sub_df['customer_id'].values.reshape((-1,))):
    if cust_id in purchase_dict:
        _list = sorted(purchase_dict[cust_id].items(), key=lambda x: x[1], reverse=True)
        _list = [y[0] for y in _list]

        if(len(_list) > 12):
            s = ' '.join(_list)
        else:
            s = ' '.join(_list + dummy_list[:(12 - len(_list)]))
    else:
        s = ' '.join(dummy_pred)

    pred_list.append(s)

benchmark['prediction'] = pred_list
logger.info(f'pred shape: {benchmark.shape}')
logger.info(f'pred sample: \n{benchmark.head()}')


# %%
benchmark.to_csv('benchmark.csv', index=False)
