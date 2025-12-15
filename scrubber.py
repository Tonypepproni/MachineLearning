import pandas as pd

class scrub:

    drop=['contact','day','month','duration','poutcome']
    job={'housemaid':0,'services':1,'admin.':2,'blue-collar':3,'technician':4,'retired':5,'management':6,'unemployed':7,'self-employed':8,'unknown':9,'entrepreneur':10,'student':11}
    marital={'married':3,'single':1,'divorced':2,'NaN':0,"unknown":0}
    education={'NaN':0,'primary':1,"secondary":2,'tertiary':3,'unknown':0}
    default={'no':1,'unknown':1,'yes':0}
    housing={'no':0,'yes':1,'unknown':0}
    loan={'no':0,'yes':1,'unknown':0}
    y={'no':0,'yes':1,'NaN':0,'unknown':0}
    


    def __init__(self):
        pass

    def clean(self,df):
        maps={'job':self.job,'marital':self.marital,'education':self.education,'default':self.default,'housing':self.housing,'loan':self.loan,'y':self.y}

        df=df.drop(columns=self.drop)
    
        for i in df.columns:
            if i in maps:
                df[i]=df[i].map(maps[i])
        print(df.columns)
        #df = pd.get_dummies(df, columns=['default'], drop_first=True)
        df=df.dropna()
        return df

scrubber=scrub()
