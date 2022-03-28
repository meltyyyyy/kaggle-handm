import os, sys
sys.path.append(os.pardir)
import numpy as np
import pandas as pd
import implicit
from scipy.sparse import coo_matrix
from implicit.evaluation import mean_average_precision_at_k
from logging import Formatter, FileHandler, getLogger
from utils.load_data import load_transaction_data, load_test_data, load_customer_data, load_article_data

LOG_DIR = '../logs/'

logger = getLogger(__name__)
log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s')

handler = FileHandler(LOG_DIR + 'als.py.log', mode='w')
handler.setLevel('DEBUG')
handler.setFormatter(log_fmt)
logger.setLevel('DEBUG')
logger.addHandler(handler)

logger.info('start')

logger.info('start loading')
transaction_df=load_transaction_data()
sub_df=load_test_data()
customer_df=load_customer_data()
article_df=load_article_data()
logger.info('finish loading')
logger.info('transaction data shape: {}'.format(transaction_df.shape))
logger.info('submission data shape: {}'.format(sub_df.shape))
logger.info('customer data shape: {}'.format(customer_df.shape))
logger.info('article data shape: {}'.format(article_df.shape))

logger.info('arrange data')
transaction_df['article_id'] = transaction_df['article_id'].astype(str)
transaction_df['t_dat'] = pd.to_datetime(transaction_df['t_dat'])
article_df['article_id'] = article_df['article_id'].astype(str)
# trying with less data
transaction_df=transaction_df[transaction_df['t_dat']>'2020-08-21']
logger.info('reduced transaction data shape: {}'.format(transaction_df.shape))

logger.info('assign ids to customers and articles')
ALL_USERS = customer_df['customer_id'].unique().tolist()
ALL_ITEMS = article_df['article_id'].unique().tolist()

user_ids = dict(list(enumerate(ALL_USERS)))
item_ids = dict(list(enumerate(ALL_ITEMS)))

user_map = {u: uidx for uidx, u in user_ids.items()}
item_map = {i: iidx for iidx, i in item_ids.items()}

transaction_df['user_id'] = transaction_df['customer_id'].map(user_map)
transaction_df['item_id'] = transaction_df['article_id'].map(item_map)

del customer_df, article_df
logger.info('create coo-matrix and csr-matrix')

row = transaction_df['user_id'].values
col = transaction_df['item_id'].values
data = np.ones(transaction_df.shape[0])
coo_train = coo_matrix((data, (row, col)), shape=(len(ALL_USERS), len(ALL_ITEMS)))

def to_user_item_coo(df):
    row = df['user_id'].values
    col = df['item_id'].values
    data = np.ones(df.shape[0])
    coo = coo_matrix((data, (row, col)), shape=(len(ALL_USERS), len(ALL_ITEMS)))
    return coo

def split_data(df, validation_days=7):
    validation_cut = df['t_dat'].max() - pd.Timedelta(validation_days)
    df_train = df[df['t_dat'] < validation_cut]
    df_val = df[df['t_dat'] >= validation_cut]
    return df_train, df_val

def get_val_matrices(df, validation_days=7):
    df_train, df_val = split_data(df, validation_days=validation_days)
    coo_train = to_user_item_coo(df_train)
    coo_val = to_user_item_coo(df_val)

    csr_train = coo_train.tocsr()
    csr_val = coo_val.tocsr()
    return {'coo_train': coo_train,
            'csr_train': csr_train,
	    'csr_val': csr_val}

def validate(matrices, factors=200, iterations=20, regularization=0.01, show_progress=True):
    coo_train, csr_train, csr_val = matrices['coo_train'], matrices['csr_train'], matrices['csr_val']
    model = implicit.als.AlternatingLeastSquares(
    factors=factors,
    iterations=iterations,
    regularization=regularization,
    random_state=42)

    model.fit(coo_train, show_progress=show_progress)
    map12 = mean_average_precision_at_k(model, csr_train, csr_val, K=12, show_progress=show_progress, num_threads=4)
    logger.info(f"Factors: {factors:>3} - Iterations: {iterations:>2} - Regularization: {regularization:4.3f} ==> MAP@12: {map12:6.5f}")
    return map12


matrices = get_val_matrices(transaction_df)

best_map12 = 0
for factors in [40, 50, 60, 100, 200, 500, 1000]:
    for iterations in [3, 12, 14, 15, 20]:
        for regularization in [0.01]:
            map12 = validate(matrices, factors, iterations, regularization, show_progress=False)
            if map12 > best_map12:
                best_map12 = map12
                best_params = {'factors': factors, 'iterations': iterations, 'regularization': regularization}
                logger.info(f"Best MAP@12 found. Updating: {best_params}")

del matrices


coo_train = to_user_item_coo(transaction_df)
csr_train = coo_train.tocsr()

def train(coo_train, factors=200, iterations=15, regularization=0.01, show_progress=True):
    model = implicit.als.AlternatingLeastSquares(
    factors=factors,
    iterations=iterations,
    regularization=regularization,
    random_state=42)

    model.fit(coo_train, show_progress=show_progress)
    return model

logger.info("best params:{}".format(best_params))

model = train(coo_train, **best_params)

def submit(model, csr_train, submission_name="submissions_als.csv"):
    preds = []
    batch_size = 2000
    to_generate = np.arange(len(ALL_USERS))
    for startidx in range(0, len(to_generate), batch_size):
        batch = to_generate[startidx : startidx + batch_size]
        ids, scores = model.recommend(batch, csr_train[batch], N=12, filter_already_liked_items=False)
        for i, userid in enumerate(batch):
            customer_id = user_ids[userid]
            user_items = ids[i]
            article_ids = [item_ids[item_id] for item_id in user_items]
            preds.append((customer_id, ' '.join(article_ids)))
    df_preds = pd.DataFrame(preds, columns=['customer_id', 'prediction'])
    df_preds.to_csv(submission_name, index=False)

    logger.info("df_preds:\n{}".format(df_preds.head()))
    logger.info("df_preds:\n{}".format(df_preds.shape))

    return df_preds

df_preds = submit(model, csr_train)

logger.info('end')

