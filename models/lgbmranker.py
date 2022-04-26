# %%
import sys
import os
sys.path.append(os.pardir)

# %%
from logs.logger import get_logger
from utils.load_data import load_submission_data, load_transaction_data, load_article_data, load_customer_data
from utils.validations import eval_sub
from preprocess.group_by_age import join_age_id
from configs.data import OUTPUT_DIR
from lightgbm import LGBMRanker
import pandas as pd

# %%
LOG_FILE = '../logs/lgbmranker.py.log'

# %%
logger = get_logger(module_name=__name__, log_file=LOG_FILE)
logger.info('start')


# %%
logger.info('load data')
tran_df = load_transaction_data()
arti_df = load_article_data()
cust_df = load_customer_data()
logger.info(f'tran_df shape: {tran_df.shape}')
logger.info(f'tran_df head: \n{tran_df.head()}')
logger.info(f'arti_df shape: {arti_df.shape}')
logger.info(f'arti_df head: \n{arti_df.head()}')
logger.info(f'cust_df shape: {cust_df.shape}')
logger.info(f'cust_df head: \n{cust_df.head()}')

# %%
test_week = tran_df.week.max()
tran_df = tran_df[tran_df.week != tran_df.week.max()]
tran_df = tran_df[tran_df.week > tran_df.week.max() - 10]

# %%
cust2weeks = tran_df.groupby('customer_id')['week'].unique()
logger.info(f'c2week shape: {cust2weeks.shape}')
logger.info(f'c2week: \n{cust2weeks.head()}')

cust2weeks_shifted_weeks = {}

for cust_id, weeks in cust2weeks.items():
    cust2weeks_shifted_weeks[cust_id] = {}
    for i in range(weeks.shape[0] - 1):
        cust2weeks_shifted_weeks[cust_id][weeks[i]] = weeks[i + 1]
    cust2weeks_shifted_weeks[cust_id][weeks[-1]] = test_week

# %%
candidates_last_purchase = tran_df.copy()

weeks = []
for i, (cust_id, week) in enumerate(
        zip(tran_df['customer_id'], tran_df['week'])):
    weeks.append(cust2weeks_shifted_weeks[cust_id][week])

candidates_last_purchase.week = weeks

# %%
mean_price = tran_df.groupby(['week', 'article_id'])['price'].mean()
sales = tran_df.groupby('week')['article_id'].value_counts().groupby('week').rank(
    method='dense', ascending=False).groupby('week').head(12).rename('bestseller_rank')
bestsellers_previous_week = pd.merge(
    sales, mean_price, on=[
        'week', 'article_id']).reset_index()
bestsellers_previous_week.week += 1
logger.info(
    f'bestseller_previous_week shape: {bestsellers_previous_week.shape}')
logger.info(
    f'bestseller_previous_week sample: \n{bestsellers_previous_week.head()}')

# %%
unique_transactions = tran_df.groupby(['week', 'customer_id']).head(
    1).drop(columns=['article_id', 'price']).copy()
candidates_bestsellers = pd.merge(
    unique_transactions,
    bestsellers_previous_week,
    on='week')
test_set_transactions = unique_transactions.drop_duplicates(
    'customer_id').reset_index(drop=True)
test_set_transactions.week = test_week
candidates_bestsellers_test_week = pd.merge(
    test_set_transactions, bestsellers_previous_week, on='week')
candidates_bestsellers = pd.concat(
    [candidates_bestsellers, candidates_bestsellers_test_week])
candidates_bestsellers.drop(columns='bestseller_rank', inplace=True)

# %%
cust_df = join_age_id(cust_df=cust_df)
cust_df.drop(columns=['age'], inplace=True)

# %%
tran_df['purchased'] = 1
data = pd.concat([tran_df, candidates_last_purchase, candidates_bestsellers])
data.purchased.fillna(0, inplace=True)
data.drop_duplicates(['customer_id', 'article_id', 'week'], inplace=True)

data = pd.merge(data, bestsellers_previous_week[[
                'week', 'article_id', 'bestseller_rank']], on=['week', 'article_id'], how='left')
data = data[data.week != data.week.min()]
data.bestseller_rank.fillna(999, inplace=True)

data = pd.merge(data, arti_df, on='article_id', how='left')
data = pd.merge(data, cust_df, on='customer_id', how='left')

data.sort_values(['week', 'customer_id'], inplace=True)
data.reset_index(drop=True, inplace=True)
logger.info(f'data info: {data.shape}')
logger.info(f'data sample: \n{data.head()}')

# %%
train_df = data[data.week != test_week]
test_df = data[data.week == test_week].drop_duplicates(
    ['customer_id', 'article_id', 'sales_channel_id']).copy()
train_baskets = train_df.groupby(['week', 'customer_id'])[
    'article_id'].count().values

# %%
columns_to_use = ['article_id', 'customer_id', 'product_type_no', 'graphical_appearance_no', 'colour_group_code', 'perceived_colour_value_id',
'perceived_colour_master_id', 'department_no', 'index_code',
'index_group_no', 'section_no', 'garment_group_no', 'FN', 'Active',
'club_member_status', 'fashion_news_frequency', 'age_id', 'postal_code', 'bestseller_rank', 'attribute']

# %%
train_X = train_df[columns_to_use]
train_y = train_df['purchased']

test_X = test_df[columns_to_use]

# %%
train_X.info()

# %%
ranker = LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    boosting_type="goss",
    n_estimators=2,
    learning_rate=0.1,
    importance_type='gain',
    verbose=10
)
ranker = ranker.fit(
    train_X,
    train_y,
    group=train_baskets,
)

# %%
for i in ranker.feature_importances_.argsort()[::-1]:
    logger.info(
        '{}:{}'.format(
            columns_to_use[i],
            ranker.feature_importances_[i] /
            ranker.feature_importances_.sum()))

# %%
test_df['preds'] = ranker.predict(test_X)

cust_id2predicted_article_ids = test_df \
    .sort_values(['customer_id', 'preds'], ascending=False) \
    .groupby('customer_id')['article_id'].apply(list).to_dict()

bestsellers_last_week = ['0' + str(arti_id) for arti_id in bestsellers_previous_week[bestsellers_previous_week.week ==
                                                                                     bestsellers_previous_week.week.max()]['article_id'].tolist()]

# %%
sub_df = load_submission_data()
logger.info(f'sub_df shape: {sub_df.shape}')
logger.info(f'sub_df sample: \n{sub_df.head()}')

# %%
preds = []
for cust_id in sub_df.customer_id:
    if cust_id in cust_id2predicted_article_ids:
        ps = [str(p) for p in cust_id2predicted_article_ids[cust_id]]
        preds.append(ps)
    else:
        preds.append(bestsellers_last_week)
preds = [' '.join(ps) for ps in preds]
sub_df.prediction = preds
logger.info(f'submission shape: {sub_df.shape}')
logger.info(f'submission sample: \n{sub_df.head()}')
sub_df.to_csv(OUTPUT_DIR + 'lgbmranker.csv', index=False)

# %%
eval_sub(sub_csv=OUTPUT_DIR + 'lgbmranker.csv', skip_cust_with_no_purchases=True)

# %%
