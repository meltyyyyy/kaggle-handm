# %%
import sys
import os
sys.path.append(os.pardir)

# %%
from memory_reduction import customer_hex_id_to_int
from categorize import Categorize
from configs.data import INPUT_DIR, OUTPUT_DIR
import pandas as pd

# %%
tran_df = pd.read_csv(INPUT_DIR + 'original/transactions_train.csv')
arti_df = pd.read_csv(INPUT_DIR + 'original/articles.csv')
cust_df = pd.read_csv(INPUT_DIR + 'original/customers.csv')

# %%
tran_df.memory_usage(deep=True)

# %%
tran_df.info(memory_usage='deep')

# %%
tran_df['t_dat'] = pd.to_datetime(tran_df['t_dat'])

# %%
valid_df = tran_df
valid_df = valid_df[valid_df['t_dat'] >= pd.to_datetime('2020-09-16')]
tran_df = tran_df[tran_df['t_dat'] <= pd.to_datetime('2020-09-15')]

# %%
valid_df = valid_df.groupby('customer_id')['article_id'].apply(list).reset_index()
valid_df = valid_df.rename({'article_id': 'prediction'}, axis=1)
valid_df['prediction'] = valid_df['prediction'].apply(lambda x: ' '.join(['0' + str(k) for k in x]))

# %%
valid_df.memory_usage(deep=True)

# %%
valid_df.info(memory_usage='deep')

# %%
valid_df.head()
valid_df.to_feather(INPUT_DIR + 'valid_sample.feather')

# %%
%%time
tran_df['customer_id'].nunique()

# %%
tran_df['customer_id'] = customer_hex_id_to_int(tran_df['customer_id'])

# %%
%%time
tran_df['customer_id'].nunique()

# %%
tran_df.memory_usage(deep=True)

# %%
tran_df.info(memory_usage='deep')

# %%
tran_df['week'] = 104 - (tran_df.t_dat.max() - tran_df.t_dat).dt.days // 7

# %%
tran_df.memory_usage(deep=True)

# %%
tran_df.info(memory_usage='deep')

# %%
tran_df['week'] = tran_df['week'].astype('int8')
tran_df['article_id'] = tran_df['article_id'].astype('int32')
tran_df['price'] = tran_df['price'].astype('float32')
tran_df['sales_channel_id'] = tran_df['sales_channel_id'].astype('int8')

# %%
tran_df.memory_usage(deep=True)

# %%
tran_df.info(memory_usage='deep')

# %%
tran_df = tran_df.sort_values(['t_dat', 'customer_id']).reset_index(drop=True)

# %%
tran_df.to_feather(INPUT_DIR + 'tran_sample.feather')

# %%
cust_df.memory_usage(deep=True)

# %%
cust_df.info(memory_usage='deep')

# %%
cust_df['customer_id'] = customer_hex_id_to_int(cust_df['customer_id'])

# %%
cust_df.memory_usage(deep=True)

# %%
cust_df.info(memory_usage='deep')

# %%
for col in ['FN', 'Active', 'age']:
    cust_df[col].fillna(-1, inplace=True)
    cust_df[col] = cust_df[col].astype('int8')

# %%
cust_df['club_member_status'] = Categorize().fit_transform(cust_df[['club_member_status']])['club_member_status']
cust_df['postal_code'] = Categorize().fit_transform(cust_df[['postal_code']])['postal_code']
cust_df['fashion_news_frequency'] = Categorize().fit_transform(cust_df[['fashion_news_frequency']])['fashion_news_frequency']

# %%
cust_df.memory_usage(deep=True)

# %%
cust_df.info(memory_usage='deep')

# %%
cust_df.to_feather(INPUT_DIR + 'cust_sample.feather')

# %%
arti_df.memory_usage(deep=True)

# %%
arti_df.info(memory_usage='deep')

# %%
arti_df['article_id'] = arti_df['article_id'].astype('int32')

# %%
arti_df.memory_usage(deep=True)

# %%
arti_df.info(memory_usage='deep')

# %%
for col in arti_df.columns:
    if arti_df[col].dtype == 'object':
        arti_df[col] = Categorize().fit_transform(arti_df[[col]])[col]

# %%
arti_df.memory_usage(deep=True)

# %%
arti_df.info(memory_usage='deep')

# %%
for col in arti_df.columns:
    if arti_df[col].dtype == 'int64':
        arti_df[col] = arti_df[col].astype('int32')

# %%
arti_df.memory_usage(deep=True)# %%

# %%
arti_df.info(memory_usage='deep')

# %%
arti_df.to_feather(INPUT_DIR + 'arti_sample.feather')

# %%
sample = 0.05
cust_sample = cust_df.sample(frac=sample, replace=False).reset_index(drop=True)
cust_sample_ids = set(cust_sample['customer_id'])
tran_sample = tran_df[tran_df["customer_id"].isin(cust_sample_ids)].reset_index(drop=True)
articles_sample_ids = set(tran_sample["article_id"])
articles_sample = arti_df[arti_df["article_id"].isin(articles_sample_ids)].reset_index(drop=True)

cust_sample.to_feather(INPUT_DIR + 'cust_sample_small.feather')
tran_sample.to_feather(INPUT_DIR + 'tran_sample_small.feather')
articles_sample.to_feather(INPUT_DIR + 'arti_sample_samll.feather')

# %%
