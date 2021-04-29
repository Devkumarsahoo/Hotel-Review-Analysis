import pandas as pd
import pickle

Review_Data=pd.read_csv("train.csv")

Review_Data.head()

Review_Data.shape

Review_Data.drop(columns=['User_ID','Browser_Used','Device_Used'],inplace=True)

Review_Data.head()

import re
import string
def text_clean(text):
  text = text.lower()
  text = re.sub('\[.*?\]','',text)
  text = re.sub('[%s]'% re.escape(string.punctuation),'',text)
  text = re.sub('\w*\d\w*','',text)
  return text
cleaned = lambda x:text_clean(x)

Review_Data['Cleaned_Description'] = pd.DataFrame(Review_Data.Description.apply(cleaned))
Review_Data.head(6)

def clean_text2(text):
  text=re.sub('[''""]','',text)
  text=re.sub('\n','',text)
  return text
cleaned2=lambda x:clean_text2(x)

Review_Data['cleaned_description_new']=pd.DataFrame(Review_Data['Cleaned_Description'].apply(cleaned2))
Review_Data.head(2)

from sklearn.model_selection import train_test_split
Independent_var=Review_Data.cleaned_description_new
Dependent_var=Review_Data.Is_Response
IV_train,IV_test,DV_train,DV_test=train_test_split(Independent_var,Dependent_var,test_size=0.1,random_state=225)
print(len(IV_train))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
tvec=TfidfVectorizer()
clf2=LogisticRegression(solver="lbfgs")
from sklearn.pipeline import Pipeline

model=Pipeline([('vectorizer',tvec),('classifier',clf2)])
model.fit(IV_train,DV_train)
from sklearn.metrics import confusion_matrix
predictions=model.predict(IV_test)
confusion_matrix(predictions,DV_test)

pickle.dump(model,open('Model.pkl','wb'))

