# %%
import sys
import os
sys.path.append(os.pardir)

# %%
from configs.data import INPUT_DIR
from utils.categorize import Categorize
import pandas as pd

# %%
tran_df = pd.read_feather(INPUT_DIR + 'tran_sample.feather')
arti_df = pd.read_feather(INPUT_DIR + 'arti_sample.feather')
cust_df = pd.read_feather(INPUT_DIR + 'cust_sample.feather')

# %%
print(arti_df['index_group_name'].unique())
print(arti_df['index_group_no'].unique())

# %%
arti_category_df = pd.DataFrame(arti_df[["article_id", "index_group_no"]])
arti_category_df.columns = ["article_id", "sex_attribute"]
arti_category_df.head()

# %%
tran_df = pd.merge(tran_df, arti_category_df, how='left', on='article_id')
tran_df.head()
# %%
cust_sex = tran_df[['customer_id', 'sex_attribute', 'article_id']].groupby(['customer_id', 'sex_attribute']).count().unstack()
cust_sex.columns = ["Woman", "Young", "Man", "Have-kids", "Sports-person"]

# %%
cust_sex["attribute"] = cust_sex.apply(lambda x : list(x[x == x.max()].index), axis=1)
print(cust_sex)

# %%
cust_sex1 = pd.DataFrame(cust_sex[["attribute"]]).reset_index()
cust_sex1["attribute"] = cust_sex1["attribute"].apply(",".join).astype(str)
del cust_sex
cust_sex1

# %%
cust_sex1.loc[~((cust_sex1["attribute"] == "Woman") |
                (cust_sex1["attribute"] == "Young") |
                (cust_sex1["attribute"] == "Man") |
                (cust_sex1["attribute"] == "Have-kids") |
                (cust_sex1["attribute"] == "Sports-person")), "attribute"] = "Woman"

# %%
print(cust_sex1.attribute.unique())

# %%
print(cust_sex1["attribute"].value_counts().sort_values(ascending=False))

# %%
cust_df = pd.merge(cust_df, cust_sex1, on='customer_id', how='left')

# %%
cust_df.isnull().sum()

# %%
cust_df['attribute'].fillna('Women', inplace=True)

# %%
cust_df.isnull().sum()

# %%
cust_df.info(memory_usage='deep')
cust_df.head()

# %%
cust_df['attribute'] = Categorize().fit_transform(cust_df[['attribute']])['attribute']

# %%
cust_df.info(memory_usage='deep')
cust_df.head()

# %%
cust_df.to_feather(INPUT_DIR + 'cust_with_sex.feather')

# %%
