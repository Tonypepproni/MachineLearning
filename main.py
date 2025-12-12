import scrubber 
import pandas as pd

df=pd.read_csv('bank-full.csv',sep=';')


df=scrubber.scrubber.clean(df)
