%matplotlib inline
import seaborn as sns
import pandas as pd
tips=pd.read_csv("stroopdata.csv")
result=sns.boxplot(data=tips)
