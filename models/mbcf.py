import logging
from logging import getLogger, FileHandler, Formatter
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd

from configs.mbcf import START_DATE
from utils.load_data import load_transaction_data, load_article_data

LOG_DIR = '../logs/'

logger = getLogger(__name__)
log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s')

handler = FileHandler(LOG_DIR + 'mbcf.py.log', mode='w')
handler.setLevel(logging.DEBUG)
handler.setFormatter(log_fmt)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

logger.info('start')

tran_df = load_transaction_data()
logger.info(f'tran_df shape: {tran_df.shape}')
logger.info(f'tran_df sample: \n{tran_df.head()}')
logger.info(f'tran_df keys: \n{tran_df[:0]}')

arti_df = load_article_data()
logger.info(f'arti_df shape: {arti_df.shape}')
logger.info(f'arti_df sample: \n{arti_df.head()}')
logger.info(f'arti_df keys: \n{arti_df[:0]}')

start_date = pd.to_datetime(START_DATE)
tran_df['t_dat'] = pd.to_datetime(tran_df['t_dat'])
tran_df = tran_df[tran_df['t_dat'] > start_date]
logger.info(f'arranged tran_df shape: {tran_df.shape}')
logger.info(f'arranged tran_df sample: \n{tran_df.head()}')

# here we reduce the memory
tran_df['customer_id'] = tran_df['customer_id'].apply(lambda x: int(x[-8:], 16)).astype('int32')
tran_df['article_id'] = tran_df['article_id'].astype('int32')
arti_df['article_id'] = arti_df['article_id'].astype('int32')

tran_df = tran_df.loc[:, ['customer_id', 'article_id']]
logger.info(f'selected tran_df: \n{tran_df.head()}')

arti_df = arti_df.loc[:, ['article_id', 'product_type_no']]
logger.info(f'selected arti_df: \n{arti_df.head()}')

cust_prod_matrix = pd.merge(tran_df, arti_df, on='article_id', how='inner')
logger.debug(f'customer product matrix: \n{cust_prod_matrix.head()}')
logger.debug('check if article_id contains Nan: \n{}'.format(cust_prod_matrix[cust_prod_matrix['article_id'].isna()]))
logger.debug('check if customer_id contains Nan: \n{}'.format(cust_prod_matrix[cust_prod_matrix['customer_id'].isna()]))
del tran_df
del arti_df

cust_prod_matrix = cust_prod_matrix.groupby(['customer_id', 'product_type_no']).size().reset_index().rename(columns={0: 'count'})
logger.info(f'customer product matrix: \n{cust_prod_matrix}')
logger.info(f'check if customer_id duplicates: \n{cust_prod_matrix.value_counts(ascending=False)}')

cust_prod_matrix = cust_prod_matrix.pivot(index='customer_id', columns='product_type_no', values='count').fillna(0)
logger.info(f'customer product matrix: \n{cust_prod_matrix}')

user_simi_matrix = cosine_similarity(cust_prod_matrix)
logger.info(f'user similarity matrix: \n{user_simi_matrix}')
logger.info(f'user_similarity_matrix shape: \n{user_simi_matrix.shape}')

class CfRec():
    def __init__(self, M, X, items, k=20, top_n=10):
        self.X = X
        self.M = M
        self.items = items
        self.k = k
        self.top_n = top_n

    def rec_user_based(self, user_id):
        ix = self.M.index.get_loc(user_id)
        u_sim = self.X[ix]
        most_similar = self.M.index[u_sim.argpartition(-(self.k + 1))[-(self.k + 1)]]
        rec_items = self.M.loc[most_similar].sort_values(ascending=False)
        seen_mask = self.M.loc[user_id].gt(0)
        seen = seen_mask.index[seen_mask].tolist()
        rec_items = rec_items.drop(seen).head(self.top_n)
        return (rec_items.index.to_frame().reset_index(drop=True).merge(self.items))

    def rec_item_based(self, item_id):
        liked = self.items.loc[self.items.movie_id.eq(item_id), 'title'].item()
        ix = self.M.columns.get_loc(liked)
        i_sim = self.X[ix]
        most_similar = self.M.columns[i_sim.argpartition(-(self.k + 1))[-(self.k + 1):]]
        return (most_similar.difference([item_id]).to_frame().reset_index(drop=True).merge(self.items).head(self.top_n))


def get_user_liked(user_item_m, movie_df, rate_df, user_id):
    ix_user_seen = user_item_m.loc[user_id] > 0
    seen_by_user = user_item_m.columns[ix_user_seen]
    return (seen_by_user.to_frame()
            .reset_index(drop=True)
            .merge(movie_df)
            .assign(user_id=user_id)
            .merge(rate_df[rate_df.user_id.eq(user_id)])
            .sort_values('rating', ascending=False)
            .head(10))
