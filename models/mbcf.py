from logging import getLogger, Formatter, FileHandler
from sklearn.metrics.pairwise import cosine_similarity
import logging
import pandas as pd

LOG_DIR = '../logs/'
RATE_DATA = '../data/input/ml-1m/ratings.csv'
USER_DATA = '../data/input/ml-1m/users.csv'
MOVIE_DATA = '../data/input/ml-1m/movies.csv'

logger = getLogger(__name__)
log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s')

handler = FileHandler(LOG_DIR + 'mbcf.py.log', mode='w')
handler.setLevel(logging.DEBUG)
handler.setFormatter(log_fmt)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

logger.info('start')
rate_df = pd.read_csv(RATE_DATA, sep='\t', encoding='latin-1', usecols=['user_id', 'movie_id', 'rating'])
user_df = pd.read_csv(USER_DATA, sep='\t', encoding='latin-1', usecols=['user_id', 'gender', 'zipcode', 'age_desc', 'occ_desc'])
movie_df = pd.read_csv(MOVIE_DATA, sep='\t', encoding='latin-1', usecols=['movie_id', 'title', 'genres'])
logger.info('rate_df shape: {}'.format(rate_df.shape))
logger.info('rate_df data samlple: \n{}'.format(rate_df.head()))
logger.info('user_df shape: {}'.format(user_df.shape))
logger.info('user_df data samlple: \n{}'.format(user_df.head()))
logger.info('movie_df shape: {}'.format(movie_df.shape))
logger.info('movie_df data samlple: \n{}'.format(movie_df.head()))

logger.info('arrange data')
user_item_matrix = rate_df.pivot('user_id', 'movie_id', 'rating').fillna(0).astype(int)
logger.info('user_item_matrix shape: {}'.format(user_item_matrix.shape))
logger.info('user_item_matrix shape: \n{}'.format(user_item_matrix.head()))


user_item_matrix = user_item_matrix.T.join(movie_df.set_index('movie_id').title).set_index('title').T.rename_axis('user_id')
logger.info(f'arranged data: \n{user_item_matrix}')

X_user = cosine_similarity(user_item_matrix)
logger.info(f'X_user: \n{X_user}')
X_item = cosine_similarity(user_item_matrix.T)
logger.info(f'X_item: \n{X_item}')


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


rec = CfRec(user_item_matrix, X_user, movie_df)
user_id = 69
user_liked = get_user_liked(user_item_matrix, movie_df, rate_df, user_id)
logger.info(f'user{user_id} liked: \n{user_liked}')

user_based_rec = rec.rec_user_based(user_id)
logger.info(f'user based recommendation\n{user_based_rec}')

user_id = 2155
user_liked = get_user_liked(user_item_matrix, movie_df, rate_df, user_id)
logger.info(f'user{user_id} liked: \n{user_liked}')

user_based_rec = rec.rec_user_based(user_id)
logger.info(f'user based recommendation\n{user_based_rec}')

rec = CfRec(user_item_matrix, X_item, movie_df)
item_id = 2021
item_based_rec = rec.rec_item_based(item_id)
logger.info(f'item based recommendation\n{item_based_rec}')

item_id = 2018
item_based_rec = rec.rec_item_based(item_id)
logger.info(f'item based recommendation\n{item_based_rec}')
