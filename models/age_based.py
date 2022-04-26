# %%
import sys, warnings, os
warnings.filterwarnings('ignore')
sys.path.append(os.pardir)
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
from tqdm import tqdm
tqdm.pandas()
from configs.data import INPUT_DIR, OUTPUT_DIR
from utils.memory_reduction import customer_hex_id_to_int
DEBUG = False

# %%
dfArticles = pd.read_csv(INPUT_DIR + 'original/articles.csv', usecols=['article_id', "product_group_name", "perceived_colour_master_name"])
dfCustomers = pd.read_csv(INPUT_DIR + 'original/customers.csv', usecols=['customer_id', 'age'])

# %%
listBin = [-1, 19, 29, 39, 49, 59, 69, 119]
dfCustomers['age_bins'] = pd.cut(dfCustomers['age'], listBin)
# %%
x = dfCustomers[dfCustomers['age_bins'].isnull()].shape[0]
print(f'{x} customer_id do not have age information.\n')

# %%
dfTransactions = pd.read_csv(
    INPUT_DIR +
    'original/transactions_train.csv',
    usecols=[
        't_dat',
        'customer_id',
        'article_id'],
    dtype={
        'article_id': 'int32',
        't_dat': 'string',
        'customer_id': 'string'})
dfTransactions['t_dat'] = pd.to_datetime(dfTransactions['t_dat'])
dfTransactions.set_index('t_dat', inplace=True)

# %%
N = 12
listUniBins = dfCustomers['age_bins'].unique().tolist()

# %%
for uniBin in listUniBins:
    df = pd.read_csv(
        INPUT_DIR +
        'original/transactions_train.csv',
        usecols=[
            't_dat',
            'customer_id',
            'article_id'],
        dtype={
            'article_id': 'int32',
            't_dat': 'string',
            'customer_id': 'string'})
    if str(uniBin) == 'nan':
        dfCustomersTemp = dfCustomers[dfCustomers['age_bins'].isnull()]
    else:
        dfCustomersTemp = dfCustomers[dfCustomers['age_bins'] == uniBin]

    dfCustomersTemp = dfCustomersTemp.drop(['age_bins'], axis=1)

    df = df.merge(
        dfCustomersTemp[['customer_id', 'age']], on='customer_id', how='inner')
    print(f'The shape of scope transaction for {uniBin} is {df.shape}. \n')

    df['customer_id'] = customer_hex_id_to_int(df['customer_id'])
    df['t_dat'] = pd.to_datetime(df['t_dat'])
    last_ts = df['t_dat'].max()

    tmp = df[['t_dat']].copy()
    tmp['dow'] = tmp['t_dat'].dt.dayofweek
    tmp['ldbw'] = tmp['t_dat'] - pd.TimedeltaIndex(tmp['dow'] - 1, unit='D')
    tmp.loc[tmp['dow'] >= 2, 'ldbw'] = tmp.loc[tmp['dow'] >= 2, 'ldbw'] + \
        pd.TimedeltaIndex(np.ones(len(tmp.loc[tmp['dow'] >= 2])) * 7, unit='D')

    df['ldbw'] = tmp['ldbw'].values

    weekly_sales = df.drop('customer_id', axis=1).groupby(
        ['ldbw', 'article_id']).count().reset_index()
    weekly_sales = weekly_sales.rename(columns={'t_dat': 'count'})

    df = df.merge(weekly_sales, on=['ldbw', 'article_id'], how='left')

    weekly_sales = weekly_sales.reset_index().set_index('article_id')

    df = df.merge(
        weekly_sales.loc[weekly_sales['ldbw'] == last_ts, ['count']],
        on='article_id', suffixes=("", "_targ"))

    df['count_targ'].fillna(0, inplace=True)
    del weekly_sales

    df['quotient'] = df['count_targ'] / df['count']

    target_sales = df.drop('customer_id', axis=1).groupby(
        'article_id')['quotient'].sum()
    general_pred = target_sales.nlargest(N).index.tolist()
    general_pred = ['0' + str(article_id) for article_id in general_pred]
    general_pred_str = ' '.join(general_pred)
    del target_sales

    purchase_dict = {}

    tmp = df.copy()
    tmp['x'] = ((last_ts - tmp['t_dat']) / np.timedelta64(1, 'D')).astype(int)
    tmp['dummy_1'] = 1
    tmp['x'] = tmp[["x", "dummy_1"]].max(axis=1)

    a, b, c, d = 2.5e4, 1.5e5, 2e-1, 1e3
    tmp['y'] = a / np.sqrt(tmp['x']) + b * np.exp(-c * tmp['x']) - d

    tmp['dummy_0'] = 0
    tmp['y'] = tmp[["y", "dummy_0"]].max(axis=1)
    tmp['value'] = tmp['quotient'] * tmp['y']

    tmp = tmp.groupby(['customer_id', 'article_id']).agg({'value': 'sum'})
    tmp = tmp.reset_index()

    tmp = tmp.loc[tmp['value'] > 0]
    tmp['rank'] = tmp.groupby("customer_id")[
        "value"].rank("dense", ascending=False)
    tmp = tmp.loc[tmp['rank'] <= 12]

    purchase_df = tmp.sort_values(
        ['customer_id', 'value'], ascending=False).reset_index(drop=True)
    purchase_df['prediction'] = '0' + \
        purchase_df['article_id'].astype(str) + ' '
    purchase_df = purchase_df.groupby('customer_id').agg(
        {'prediction': sum}).reset_index()
    purchase_df['prediction'] = purchase_df['prediction'].str.strip()
    purchase_df = pd.DataFrame(purchase_df)

    sub = pd.read_csv(INPUT_DIR + 'original/sample_submission.csv',
                      usecols=['customer_id'],
                      dtype={'customer_id': 'string'})

    numCustomers = sub.shape[0]

    sub = sub.merge(
        dfCustomersTemp[['customer_id', 'age']], on='customer_id', how='inner')

    sub['customer_id2'] = customer_hex_id_to_int(sub['customer_id'])

    sub = sub.merge(
        purchase_df,
        left_on='customer_id2',
        right_on='customer_id',
        how='left',
        suffixes=(
            '',
            '_ignored'))

    sub['prediction'] = sub['prediction'].fillna(general_pred_str)
    sub['prediction'] = sub['prediction'] + ' ' + general_pred_str
    sub['prediction'] = sub['prediction'].str.strip()
    sub['prediction'] = sub['prediction'].str[:131]
    sub = sub[['customer_id', 'prediction']]
    sub.to_csv(
        OUTPUT_DIR +
        'temp/submission_' +
        str(uniBin) +
        '.csv',
        index=False)
    print(f'Saved prediction for {uniBin}. The shape is {sub.shape}. \n')
    print('-' * 50)

# %%
for i, uniBin in enumerate(listUniBins):
    dfTemp = pd.read_csv(
        OUTPUT_DIR +
        'temp/submission_' +
        str(uniBin) +
        '.csv')
    if i == 0:
        dfSub = dfTemp
    else:
        dfSub = pd.concat([dfSub, dfTemp], axis=0)

assert dfSub.shape[
    0] == numCustomers, f'The number of dfSub rows is not correct. {dfSub.shape[0]} vs {numCustomers}.'

dfSub.to_csv(OUTPUT_DIR + 'age_based.csv', index=False)

# %%
