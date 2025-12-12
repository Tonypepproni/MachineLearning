import pandas as pd

class scrub:

    drop=['contact','day','month','duration','poutcome']
    job={'housemaid':0,'services':1,'admin.':2,'blue-collar':3,'technician':4,'retired':5,'management':6,'unemployed':7,'self-employed':8,'unknown':9,'entrepreneur':10,'student':11}
    marital={'married':0,'single':1,'divorced':2,'unkown':3}
    education={'unkown':0,'primary':1,"secondary":2,'tertiary':3}
    default={'no':0,'unknown':1,'yes':2}
    housing={'no':0,'yes':1,'unknown':2}
    loan={'no':0,'yes':1,'unknown':2}
    y={'no':0,'yes':1,'unkown':2}
    


    def __init__(self):
        pass

    def clean(self,df):
        maps={'job':self.job,'marital':self.marital,'education':self.education,'default':self.default,'housing':self.housing,'loan':self.loan,'y':self.y}

        df=df.drop(columns=self.drop)
    
        for i in df.columns:
            if i in maps:
                df[i]=df[i].map(maps[i])

        print(df.head(5))
        return df

scrubber=scrub()
