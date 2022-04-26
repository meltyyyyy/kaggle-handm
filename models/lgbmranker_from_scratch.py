# %%
import sys
import os
sys.path.append(os.pardir)

# %%
from configs.data import INPUT_DIR
import pandas as pd

# %%
tran_df = pd.read_feather(INPUT_DIR + 'tran_sample.feather')
arti_df = pd.read_feather(INPUT_DIR + 'arti_sample.feather')
cust_df = pd.read_feather(INPUT_DIR + 'cust_sample.feather')

# %%
tran_df = tran_df.query('95 < week <= 103')

# %%
tran_df['bestseller'] = tran_df['article_id'].value_counts().rename('bestseller')

# %%
sales = tran_df.groupby('week')['article_id'].value_counts().groupby('week').rank(
    method='dense', ascending=False).groupby('week').head(12).rename('bestseller_rank')

