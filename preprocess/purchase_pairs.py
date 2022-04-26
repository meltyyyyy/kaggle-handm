# %%
import sys
import os
sys.path.append(os.pardir)

# %%
import numpy as np
import pandas as pd
from configs.data import INPUT_DIR

# %%


# %%
df = pd.read_feather(INPUT_DIR + 'tran_sample.feather')
df.info()

# %%
vc = df.article_id.value_counts()
pairs = {}
for j, i in enumerate(vc.index.values[1000:1032]):
    USERS = df.loc[df.article_id == i.item(), 'customer_id'].unique()
    vc2 = df.loc[(df.customer_id.isin(USERS)) & (
        df.article_id != i.item()), 'article_id'].value_counts()
    pairs[i.item()] = [vc2.index[0], vc2.index[1], vc2.index[2]]

# %%
pairs

# %%
np.save(INPUT_DIR + 'purchase_pair.npy', pairs)
