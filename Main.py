import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

a=pd.read_csv("APA-DDoS-Dataset.csv")
print(a)

le=LabelEncoder()
##a['frame.time'] = le.fit_transform(a['frame.time'])
a['ip.dst'] = le.fit_transform(a['ip.dst'])
a['ip.src'] = le.fit_transform(a['ip.src'])
a['Label'] = le.fit_transform(a['Label'])

##features
X=a.drop(['Label','frame.time'],axis=1)
print(X)

##labels
Y=a['Label']
print(Y)

##traing and testing part 
x_train,x_test,y_train,y_test = train_test_split(X,Y,shuffle=True,test_size=0.4, random_state=0)

##Algorithm Implementation

##Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(x_train, y_train)  #train the data
y_pred=NB.predict(x_test)
##print(y_pred)
##print(y_test)
print('Naive Bayes ACCURACY is', accuracy_score(y_test,y_pred))

##Python Flask
from flask import *
import pickle
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("browser1.html")  
@app.route('/login',methods = ['POST'])  
def login():  
      uname=request.form['files']
      rr=pd.read_csv(uname)
      le=LabelEncoder()
      rr['ip.dst'] = le.fit_transform(rr['ip.dst'])
      rr['ip.src'] = le.fit_transform(rr['ip.src'])
      
      type(rr)
      y_pre=NB.predict(rr)
      if y_pre[0]==0:
          return render_template('index1.html')
      elif y_pre[0]==1:
          return render_template('index2.html')
      elif y_pre[0]==2:
          return render_template('index3.html')
       
if __name__ == '__main__':
   app.run()  









