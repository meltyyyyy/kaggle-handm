# %%
from load_data import load_transaction_data, load_article_data, load_customer_data
from memory_reduction import customer_hex_id_to_int, article_id_int_to_str
import numpy as np
import pandas as pd

# %%
tran_df = load_transaction_data()
arti_df = load_article_data()
cust_df = load_customer_data()

# %%
tran_df.memory_usage(deep=True)

# %%
tran_df.info(memory_usage='deep')

# %%t
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
tran_df['t_dat'] = pd.to_datetime(tran_df['t_dat'])

# %%
tran_df['week'] = tran_df['t_dat'].dt.isocalendar().week.astype('int8')

# %%
tran_df.memory_usage(deep=True)

# %%
tran_df.info(memory_usage='deep')

# %%
tran_df['article_id'] = tran_df['article_id'].astype('int32')
tran_df['price'] = tran_df['price'].astype('float32')
tran_df['sales_channel_id'] = tran_df['sales_channel_id'].astype('int8')

# %%
tran_df.memory_usage(deep=True)

# %%
tran_df.info(memory_usage='deep')

# %%
