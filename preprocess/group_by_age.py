# %%
import sys
import os
sys.path.append(os.pardir)

# %%
from configs.data import INPUT_DIR
import pandas as pd

# %%
cust_df = pd.read_feather(INPUT_DIR + 'cust_sample.feather')

# %%
cust_df.info(memory_usage='deep')

# %%
