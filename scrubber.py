import pandas as pd

class scrub:

    job={'housemaid':0,'services':1,'admin.':2,'blue-collar':3,'technician':4,'retired':5,'management':6,'unemployed':7,'self-employed':8,'unknown':9,'entrepreneur':10,'student':11}
    marital={'married':0,'single':1,'divorced':2,'unkown':3}
    education={'basic.4y':0,'high.school':1,'basic.6y':2,'basic.9y':3,'professional.course':4,'unkown':5,'university.degree':6,'illiterate':7}
    default={'no':0,'unknown':1,'yes':2}
    housing={'no':0,'yes':1,'unknown':2}
    loan={'no':0,'yes':1,'unknown':2}
    


    def __init__(self,data):
        pass

scrubber=scrub()