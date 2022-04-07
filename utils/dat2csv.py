import pandas as pd

target = [
    'ml-1m/ratings',
    'ml-1m/users',
    'ml-1m/movies'
]

extension = 'dat'

for t in target:
    df = pd.read_csv('./data/input/' + t + '.' + extension, sep='\t', header=None, encoding="latin-1")
    print(df.head())
    df.to_csv('./data/input/' + t + '.csv')
