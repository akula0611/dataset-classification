import sys
print("python: {}".format(sys.version))
import scipy
print("scipy: {}".format(scipy.__version__))
!pip install scipy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
url='https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/639388c2cbc2120a14dcf466e85730eb8be498bb/iris.csv'
dataset=pd.read_csv(url)
dataset.head()
dataset.hist()
plt.show()
x=dataset.values[:,:4]
y=dataset.values[:,4]
print(x[1],"--",y[1])
x_train,x_validation,y_train,y_validation=train_test_split(x,y,test_size=0.2,random_state=0)
#building models
models=[]
models.append(('LR',LogisticRegression(solver='liblinear',multi_class='ovr')))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC(gamma='auto')))
results=[]
names=[]
for name,model in models:
    kfold=StratifiedKFold(n_splits=10,random_state=1)
    cv_results=cross_val_score(model,x_train,y_train,cv=kfold,scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(name,":",cv_results.mean()," ",cv_results.std())
    plt.boxplot(results,labels=names)
plt.title('Algorithm comparison')
plt.show()
#from the boxplot i got LinearDiscriminantAnalysis has the highest accuracy
model=LinearDiscriminantAnalysis()
model.fit(x_train,y_train)
predictions=model.predict(x_validation)
print(accuracy_score(y_validation,predictions))
print(confusion_matrix(y_validation,predictions))
print(classification_report(y_validation,predictions))
