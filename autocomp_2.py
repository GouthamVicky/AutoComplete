import pandas as pd
import numpy as np
import re

def predict_company(words):
    words = words.upper()
    df = pd.read_csv("companyname.csv")
    result =df[df['Company Name'].str.match(r".*"+words+r".*")== True]
    res=result["Company Name"].tolist()
    #print(res)
    if res==[]:
        new =df['Company Name'].str.replace(' ', '')
        df['without_space']=new
        result =df[df['without_space'].str.match(r".*"+words+r".*")== True]
        res=result["Company Name"].tolist()       
    return res

words ="vakilsea"
print(predict_company(words))

