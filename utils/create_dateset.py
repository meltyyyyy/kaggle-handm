# %%
from load_data import load_transaction_data, load_article_data, load_customer_data
from memory_reduction import customer_hex_id_to_int
import numpy as np

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
