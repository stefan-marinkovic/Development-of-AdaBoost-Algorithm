import pandas as pd
import numpy as np
import math
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

#%% Priprema podataka

data = pd.read_csv('data/drugY.csv')
X = data.drop('Drug',axis=1)
y = data['Drug']*2-1

X = pd.get_dummies(X)   # dummy coding: pretvaranje kategorickih u numericke
n,m = X.shape

#%% Algoritam: Ucenje
alfas = pd.Series(np.array([1/n]*n), index=data.index)   # tezine instanci: za prvi model su sve instance podjednako vazne

ensemble_size = 5
ensemble = []
weights = np.zeros(ensemble_size)   # tezine modela u ansamblu, koje odredjuju jacinu prilikom glasanja
probs=np.zeros(ensemble_size)
probs=pd.DataFrame()

for t in range(ensemble_size):
	alg = GaussianNB()   # "slabi"/jednostavni model
	alg = DecisionTreeClassifier(max_depth=2)   # "slabi"/jednostavni model
	model = alg.fit(X,y, sample_weight=alfas)
	predictions = model.predict(X)
	probs[t]=np.array(pd.DataFrame(model.predict_proba(X)).max(axis=1))
	error = (predictions!=y).astype(int)      # greska i-tog modela u predvidjanju
	weighted_error = (error*alfas).sum()       # ukupna (otezana) greska sa tezinama instanci
	w = 1/2 * math.log((1-weighted_error)/weighted_error)   # preracunavanje tezina modela

	ensemble.append(model)
	weights[t] = w

	factor = np.exp(-w*predictions*y)
	alfas = alfas * factor   # preracunavanje novih tezina instanci, po formuli
	
	z = alfas.sum()   # norma za normalizaciju
	alfas = alfas/z
    

#%% Algoritam: Predvidjanje i Evaluacija
predictions = pd.DataFrame([model.predict(X) for model in ensemble]).T
predictions['ensemble'] = np.sign(predictions.dot(weights))

print((predictions.add(y, axis=0).abs()/2).mean()) # tacnost pojedinacnih modela i ansambla (komplementarnost)
pom=(predictions.add(predictions['ensemble'], axis=0).abs()/2).drop('ensemble',axis=1)
sum_weights=sum(weights)
verovatnoce_tacnih=pom*probs
verovatnoce_tacnih.apply(lambda x:x.dot(weights)/sum_weights,axis=1)


pom.apply(lambda x:x.dot(weights)/sum_weights,axis=1)


alg = GaussianNB()
alg.fit(X,y)
pom_pred=alg.predict_proba(X)
pom_pred=pd.DataFrame(alg.predict_proba(X)).max(axis=1)
pom_pred.max(axis=1)

np.array(pom_pred)

pom_pred.apply(lambda x: x/x.sum)

alg = DecisionTreeClassifier(max_depth=2)
alg.fit(X,y)
pom_pred=alg.predict_proba(X,)
