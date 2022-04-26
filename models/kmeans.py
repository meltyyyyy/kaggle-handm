# %%
import sys, warnings, os
warnings.filterwarnings('ignore')
sys.path.append(os.pardir)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
from tqdm import tqdm
tqdm.pandas()
from configs.data import INPUT_DIR, OUTPUT_DIR
from utils.memory_reduction import customer_hex_id_to_int
from sklearn.cluster import KMeans
from sklearn import preprocessing
DEBUG = False

# %%


class Clustering_HandM():
    def customers_preprocessing(
            self,
            customers,
            dropcol=['postal_code'],
            **kwargs):
        customers = customers.drop(dropcol, axis=1)
        customers_col = list(customers.columns)

        if 'fashion_news_frequency' in customers_col:
            customers['fashion_news_frequency'] = customers['fashion_news_frequency'].replace(
                'NONE', 'None')
            customers['fashion_news_frequency'] = customers['fashion_news_frequency'].replace(
                {np.nan: 0, 'None': 0, 'Monthly': 1, 'Regularly': 2})

        if 'club_member_status' in customers_col:
            customers['club_member_status'] = customers['club_member_status'].replace(
                {np.nan: 0, 'PRE-CREATE': 1, 'ACTIVE': 2, 'LEFT CLUB': -1})

        if 'age' in customers_col:
            customers['age'] = customers['age'].fillna(-1)

        if 'FN' in customers_col:
            customers['FN'] = customers['FN'].fillna(0)

        if 'Active' in customers_col:
            customers['Active'] = customers['Active'].fillna(0)

            print(f'###NULL DESCRIPTION###\n{customers.isnull().sum()}')

        return customers

    def clustering(
            self,
            df,
            predcol,
            usecol,
            normmethod='StandardScaler',
            clusters=12,
            DEBUG=False):

        X = np.array(df[usecol])

        if normmethod == 'StandardScaler':
            nm = preprocessing.StandardScaler()
            X = nm.fit_transform(X)
        elif normmethod == 'minMax':
            nm = preprocessing.MinMaxScaler()
            X = nm.fit_transform(X)
        print(f'NormarlizationMethod:{normmethod}')

        km = KMeans(n_clusters=clusters, random_state=2022)
        km.fit(X)
        print('Distortion: %.2f' % km.inertia_)

        pred = km.labels_
        df_pred = pd.DataFrame(pred, columns=['pred'])
        df_pred = pd.concat([df, df_pred], axis=1)

        df_norm = pd.DataFrame(X, columns=usecol)
        print(df_norm.describe())

        if DEBUG:
            df_norm = pd.concat([df[predcol], df_norm], axis=1)
            return df_pred, df_norm
        else:
            return df_pred


# %%
customers = pd.read_csv(INPUT_DIR + 'original/customers.csv')

clst = Clustering_HandM()
customers = clst.customers_preprocessing(customers)

usecol = [
    'club_member_status',
    'fashion_news_frequency',
    'age',
    'FN',
    'Active']
predcol = ['customer_id']

dfCustomers = clst.clustering(customers, predcol=predcol, usecol=usecol, )

# %%
listBin = [-1, 19, 29, 39, 49, 59, 69, 119]
dfCustomers['age_bins'] = pd.cut(dfCustomers['age'], listBin)

# %%
pd.crosstab(dfCustomers['pred'], dfCustomers['age_bins'])

# %%
dfCustomers = dfCustomers.drop(['age_bins'], axis=1)

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
dfTransactions.head()

# %%
dfRecent = dfTransactions.loc['2020-09-01' : '2020-09-21']
dfRecent.head()

# %%
dfRecent = dfRecent.merge(dfCustomers[['customer_id', 'pred']], on='customer_id', how='inner')
dfRecent.head()

# %%
dfRecent = dfRecent.groupby(['pred', 'article_id']).count().reset_index().rename(columns={'customer_id': 'counts'})
listUniBins = dfRecent['pred'].unique().tolist()

dict100 = {}
for uniBin in listUniBins:
    dfTemp = dfRecent[dfRecent['pred'] == uniBin]
    dfTemp = dfTemp.sort_values(by='counts', ascending=False)
    dict100[uniBin] = dfTemp.head(100)['article_id'].values.tolist()

df100 = pd.DataFrame([dict100]).T.rename(columns={0:'top100'})

# %%
for index in df100.index:
    df100[index] = [len(set(df100.at[index, 'top100']) & set(df100.at[x, 'top100']))/100 for x in df100.index]

df100 = df100.drop(columns='top100')
plt.figure(figsize=(10, 6))
sns.heatmap(df100, annot=True, cbar=False)

# %%
N = 12
listUniBins = dfCustomers['pred'].unique().tolist()

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
        dfCustomersTemp = dfCustomers[dfCustomers['pred'].isnull()]
    else:
        dfCustomersTemp = dfCustomers[dfCustomers['pred'] == uniBin]

    dfCustomersTemp = dfCustomersTemp.drop(['pred'], axis=1)

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

print('Finished.\n')
print('='*50)

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

dfSub.to_csv(OUTPUT_DIR + 'kmeans.csv', index=False)
