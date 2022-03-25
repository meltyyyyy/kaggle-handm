mport pandas as pd

target = [
    'train',
    'test',
]

extension = 'csv'

for t in target:
(pd.read_csv('./data/input/' + t + '.' + extension, encoding="utf-8"))\
	.to_feather('./data/input/' + t + '.feather')
