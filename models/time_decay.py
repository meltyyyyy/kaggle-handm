# %%
import sys
import os
sys.path.append(os.pardir)

# %%
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import datetime
from configs.data import INPUT_DIR, OUTPUT_DIR

# %%
data = pd.read_csv(INPUT_DIR + "original/transactions_train.csv", dtype={'article_id':str})

# %%
print("All Transactions Date Range: {} to {}".format(data['t_dat'].min(), data['t_dat'].max()))

data["t_dat"] = pd.to_datetime(data["t_dat"])
train1 = data.loc[(data["t_dat"] >= datetime.datetime(2020,9,8)) & (data['t_dat'] < datetime.datetime(2020,9,16))]
train2 = data.loc[(data["t_dat"] >= datetime.datetime(2020,9,1)) & (data['t_dat'] < datetime.datetime(2020,9,8))]
train3 = data.loc[(data["t_dat"] >= datetime.datetime(2020,8,23)) & (data['t_dat'] < datetime.datetime(2020,9,1))]
train4 = data.loc[(data["t_dat"] >= datetime.datetime(2020,8,15)) & (data['t_dat'] < datetime.datetime(2020,8,23))]

val = data.loc[data["t_dat"] >= datetime.datetime(2020,9,16)]

# %%
positive_items_per_user1 = train1.groupby(['customer_id'])['article_id'].apply(list)
positive_items_per_user2 = train2.groupby(['customer_id'])['article_id'].apply(list)
positive_items_per_user3 = train3.groupby(['customer_id'])['article_id'].apply(list)
positive_items_per_user4 = train4.groupby(['customer_id'])['article_id'].apply(list)

# %%
train = pd.concat([train1, train2], axis=0)
train['pop_factor'] = train['t_dat'].apply(lambda x: 1/(datetime.datetime(2020,9,16) - x).days)
popular_items_group = train.groupby(['article_id'])['pop_factor'].sum()

_, popular_items = zip(*sorted(zip(popular_items_group, popular_items_group.keys()))[::-1])

train['pop_factor'].describe()

# %%
def apk(actual, predicted, k=12):
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=12):
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

# %%
positive_items_val = val.groupby(['customer_id'])['article_id'].apply(list)

# %%
val_users = positive_items_val.keys()
val_items = []

for i,user in tqdm(enumerate(val_users)):
    val_items.append(positive_items_val[user])

print("Total users in validation:", len(val_users))

# %%
from collections import Counter
outputs = []
cnt = 0

popular_items = list(popular_items)

for user in tqdm(val_users):
    user_output = []
    if user in positive_items_per_user1.keys():
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user1[user]).most_common()}
        user_output += list(most_common_items_of_user.keys())[:12]
    if user in positive_items_per_user2.keys():
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user2[user]).most_common()}
        user_output += list(most_common_items_of_user.keys())[:12]
    if user in positive_items_per_user3.keys():
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user3[user]).most_common()}
        user_output += list(most_common_items_of_user.keys())[:12]
    if user in positive_items_per_user4.keys():
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user4[user]).most_common()}
        user_output += list(most_common_items_of_user.keys())[:12]

    user_output += list(popular_items[:12 - len(user_output)])
    outputs.append(user_output)

print("mAP Score on Validation set:", mapk(val_items, outputs))

# %%
train1 = data.loc[(data["t_dat"] >= datetime.datetime(2020,9,16)) & (data['t_dat'] < datetime.datetime(2020,9,23))]
train2 = data.loc[(data["t_dat"] >= datetime.datetime(2020,9,8)) & (data['t_dat'] < datetime.datetime(2020,9,16))]
train3 = data.loc[(data["t_dat"] >= datetime.datetime(2020,8,31)) & (data['t_dat'] < datetime.datetime(2020,9,8))]
train4 = data.loc[(data["t_dat"] >= datetime.datetime(2020,8,23)) & (data['t_dat'] < datetime.datetime(2020,8,31))]

positive_items_per_user1 = train1.groupby(['customer_id'])['article_id'].apply(list)
positive_items_per_user2 = train2.groupby(['customer_id'])['article_id'].apply(list)
positive_items_per_user3 = train3.groupby(['customer_id'])['article_id'].apply(list)
positive_items_per_user4 = train4.groupby(['customer_id'])['article_id'].apply(list)

train = pd.concat([train1, train2], axis=0)
train['pop_factor'] = train['t_dat'].apply(lambda x: 1/(datetime.datetime(2020,9,23) - x).days)
popular_items_group = train.groupby(['article_id'])['pop_factor'].sum()

_, popular_items = zip(*sorted(zip(popular_items_group, popular_items_group.keys()))[::-1])

user_group = pd.concat([train1, train2, train3, train4], axis=0).groupby(['customer_id'])['article_id'].apply(list)

# %%
submission = pd.read_csv(INPUT_DIR + "original/sample_submission.csv")

# %%
from collections import Counter
outputs = []
cnt = 0

for user in tqdm(submission['customer_id']):
    user_output = []
    if user in positive_items_per_user1.keys():
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user1[user]).most_common()}
        user_output += list(most_common_items_of_user.keys())[:12]
    if user in positive_items_per_user2.keys():
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user2[user]).most_common()}
        user_output += list(most_common_items_of_user.keys())[:12 - len(user_output)]
    if user in positive_items_per_user3.keys():
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user3[user]).most_common()}
        user_output += list(most_common_items_of_user.keys())[:12 - len(user_output)]
    if user in positive_items_per_user4.keys():
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user4[user]).most_common()}
        user_output += list(most_common_items_of_user.keys())[:12 - len(user_output)]

    user_output += list(popular_items[:12 - len(user_output)])
    outputs.append(user_output)

str_outputs = []
for output in outputs:
    str_outputs.append(" ".join([str(x) for x in output]))

# %%
submission['prediction'] = str_outputs
submission.to_csv(OUTPUT_DIR + "time_decay.csv", index=False)
