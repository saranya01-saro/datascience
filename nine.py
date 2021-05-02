

import pandas as pd
df=pd.read_csv('Admission_Predict_Ver1.1.csv',index_col=0)
df.columns = [c.replace(' ', '_') for c in df.columns]


import numpy as np 
median = df.loc[df['Chance_of_Admit_']>=0.35, 'Chance_of_Admit_'].median()
df.loc[df.Chance_of_Admit_<= 0.35, 'Chance_of_Admit_'] = np.nan
df.fillna(median,inplace=True)
df['Chance_of_Admit_'] = np.where((df.Chance_of_Admit_ >0.6),1,df.Chance_of_Admit_)
df['Chance_of_Admit_'] = np.where((df.Chance_of_Admit_ <=0.6),0,df.Chance_of_Admit_)


x=df.iloc[:,0:-1]
y=df.iloc[:,-1]


from sklearn import linear_model
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split 
from sklearn.metrics import roc_curve
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1) 
reg = linear_model.LogisticRegression() 
reg.fit(X_train, y_train) 
pred_prob1 = reg.predict(X_test) 
# variance score: 1 means perfect prediction 
print("Logistic Regression:")
print('Accuracy: {}'.format(reg.score(X_test, y_test)*100)) 
print("Classification Report:")
print(classification_report(y_test,reg.predict(X_test)))

fpr,tpr,thresholds = roc_curve(y_test, reg.predict(X_test))
import statsmodels.api as sm

log_clf =sm.Logit(y_train,X_train)

classifier = log_clf.fit()

y_pred = classifier.predict(X_test)

print(classifier.summary2())




from sklearn.tree import DecisionTreeClassifier 
  
# create a regressor object
reg = DecisionTreeClassifier(random_state = 0) 
  
# # fit the regressor with X and Y data
reg.fit(X_train, y_train)
pred_prob2= reg.predict(X_test) 
print("DecisionTreeClassifier:")
print('Accuracy: {}'.format(reg.score(X_test, y_test)*100))  
print("Classification Report:")
print(classification_report(y_test,reg.predict(X_test)))
fpr1,tpr1,thresholds1 = roc_curve(y_test, reg.predict(X_test))

from sklearn.ensemble import RandomForestClassifier

#  # create regressor object
reg = RandomForestClassifier(random_state=0,max_depth=1)

# # fit the regressor with x and y data
reg.fit(X_train, y_train) 
pred_prob3 = reg.predict(X_test) 
print("RandomForestClassifier:")
print('Accuracy score: {}'.format(reg.score(X_test, y_test)*100)) 
print("Classification Report:")
print(classification_report(y_test,reg.predict(X_test)))

fpr2,tpr2,thresholds2 = roc_curve(y_test, reg.predict(X_test))


import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')
plt.plot(fpr, tpr, linestyle='--',color='blue', label='Logistic Regression')
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='DecisionTreeClassifier')
plt.plot(fpr2, tpr2, linestyle='--',color='green', label='RandomForestClassifier')








