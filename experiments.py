import sys
from emm import EMM
import warnings
import pandas as pd

sys.path.append('emm')
warnings.filterwarnings('ignore', category=DeprecationWarning)

df = pd.read_pickle('df_stocks.pkl')
df = df.drop(['Ticker', 'MACD', 'Price Change'], axis=1)
target_columns = ['Ticker_diff']
target_col2 = ['City', 'City']
clf = EMM.EMM(width=8, depth=4, evaluation_metric='correlation', strategy='maximize')
clf.search(df, target_cols=target_columns)