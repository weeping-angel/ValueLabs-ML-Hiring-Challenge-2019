'''
LR-RF Model
'''
import pandas as pd
import numpy as np
import operator
from functools import reduce
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
import pickle

data = pd.read_csv('Train_vec.csv')
data = np.array(data)

x_train = data[:,:9]
y_train = data[:,9]

print('x_train[1] : ', x_train[1])
print('y_train[1] : ', y_train[1])

lr = LR(random_state=1, solver='lbfgs', C=1.0, multi_class='ovr')
lr.fit(x_train, y_train)

preds = lr.predict(x_train)
preds_prob = lr.predict_proba(x_train)
lr_score = lr.score(x_train, y_train)

print('LR - Predictions ('+str(len(preds))+') : ', preds)
#print('LR - Prediction Probability : ', preds_prob)
print('LR - Scores : ', lr_score)

#print('Are all predictions of LR are correct : ', reduce(operator.and_, y_train == preds))

idxs=[x for x in range(0, len(preds)) if preds[x]==1]

x_train_2 = x_train#[idxs]
y_train_2 = y_train#[idxs]

rf = RF(n_estimators=500, max_depth=30, min_weight_fraction_leaf=0.1, random_state=None)
#rf = pickle.load(open('rf_model','rb'))
rf.fit(x_train_2, y_train_2)

preds = rf.predict(x_train_2)
preds_prob = rf.predict_proba(x_train_2)
rf_score = rf.score(x_train_2, y_train_2)
print('RF - Predictions ('+ str(len(preds)) +') : ', preds)
#print('RF - Prediction Probability : ', preds_prob)
print('RF - Score : ', rf.score(x_train_2, y_train_2))
#print('Are all predictions of RF are correct : ', reduce(operator.and_, y_train_2 == preds))


pickle.dump(lr, open('lr_model', 'wb'))
pickle.dump(rf, open('rf_model', 'wb'))



