import pandas as pd

target = [
    'transactions_train',
    'articles',
]

extension = 'csv'

for t in target:
    df = pd.read_csv('./data/input/' + t + '.' + extension, encoding="utf-8")
    df.to_feather('./data/input/' + t + '.feather')
