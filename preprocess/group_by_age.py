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
cust_df['age'].plot.hist(bins=50)

# %%


def create_age_df():
    age_group = pd.DataFrame(columns=["age", "age_id"])
    temp_group = pd.DataFrame({"age": [-1], "age_id": [-1]})
    age_group = age_group.append(temp_group)

    age_id = 0
    age = 16

    for i in range(53):
        if age < 30:
            temp_group = pd.DataFrame(
                {"age": [age, age + 1], "age_id": [age_id, age_id]})
            age_group = age_group.append(temp_group)
            age += 2
            age_id += 1
        elif age < 60:
            temp_group = pd.DataFrame({"age": [age,
                                               age + 1,
                                               age + 2,
                                               age + 3,
                                               age + 4],
                                       "age_id": [age_id,
                                                  age_id,
                                                  age_id,
                                                  age_id,
                                                  age_id]})
            age_group = age_group.append(temp_group)
            age += 5
            age_id += 1
        else:
            temp_group = pd.DataFrame({"age": [age], "age_id": [age_id]})
            age_group = age_group.append(temp_group)
            age += 1

    age_group['age'] = age_group['age'].astype('int8')
    age_group['age_id'] = age_group['age_id'].astype('int8')
    age_group.reset_index(drop=True, inplace=True)
    return age_group

# %%


def join_age_id(cust_df):
    age_group = create_age_df()
    cust_df = pd.merge(cust_df, age_group, on="age", how="left")
    return cust_df

# %%
age_group = create_age_df()

# %%
cust_df = join_age_id(cust_df=cust_df)

# %%
cust_df.info()

# %%
