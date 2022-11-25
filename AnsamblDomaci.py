import pandas as pd
import numpy as np
import math
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import random
from copy import deepcopy


data = pd.read_csv('data/drugY.csv')
X = data.drop('Drug',axis=1)
y = data['Drug']*2-1

#Videti gde ubaciti tezine
#%%
class Ensable:
    
        
    
    def learn(self,X,y,ensemble_size,algoritams,learning_rate):
        X = pd.get_dummies(X)
        n,m = X.shape
        alfas = pd.Series(np.array([1/n]*n), index=X.index)
        self.ensemble = []
        self.weights = np.zeros(ensemble_size)
        #self.probs=pd.DataFrame()

        for t in range(ensemble_size):
        	#alg = GaussianNB()   # "slabi"/jednostavni model
       # 	alg_name=algoritams[random.randint(0,len(algoritams)-1)]
        #	alg = GaussianNB() if alg_name=='NB' else DecisionTreeClassifier(max_depth=2)
        #	algoritmas_pom=algoritams.copy()
        	alg=deepcopy(algoritams[random.randint(0,len(algoritams)-1)])
        	model = alg.fit(X,y, sample_weight=alfas)
        	predictions = model.predict(X)
        	#self.probs[t]=np.array(pd.DataFrame(model.predict_proba(X)).max(axis=1))
        	error = (predictions!=y).astype(int)      # greska i-tog modela u predvidjanju
        	weighted_error = (error*alfas).sum()       # ukupna (otezana) greska sa tezinama instanci
        	w = 1/2 * math.log((1-weighted_error)/weighted_error)   # preracunavanje tezina modela
        	self.ensemble.append(model)
        	self.weights[t] = w
        	factor = np.exp(-w*predictions*y*learning_rate) #gde staviti learning rate
        	alfas = alfas * factor   # preracunavanje novih tezina instanci, po formuli        	
        	z = alfas.sum()   # norma za normalizaciju
        	alfas = alfas/z
            
    def predict(self,data):   #videti gde ce ici probs
        X=pd.get_dummies(data)
        predictions = pd.DataFrame([model.predict(X) for model in self.ensemble]).T        
        predictions['ensemble'] = np.sign(predictions.dot(self.weights))
        probs=pd.DataFrame([model.predict_proba(X).max(axis=1) for model in self.ensemble]).T        
        print((predictions.add(y, axis=0).abs()/2).mean()) # tacnost pojedinacnih modela i ansambla (komplementarnost)
        pom=(predictions.add(y, axis=0).abs()/2).drop('ensemble',axis=1)
        sum_weights=sum(self.weights)
        #verovatnoce_tacnih=pom*self.probs
        verovatnoce_tacnih=pom*probs
        return verovatnoce_tacnih.apply(lambda x:x.dot(self.weights)/sum_weights,axis=1),predictions['ensemble']
    

#%%
algoritams=[GaussianNB(),DecisionTreeClassifier(max_depth=2)]
#algoritams=['NB','Tree']
#alg_name=algoritams[random.randint(0,len(algoritams)-1)]
#alg = GaussianNB() if alg_name=='NB' else DecisionTreeClassifier

#algoritams.copy()
#alg.ensemble
alg=Ensable()
alg.learn(X, y, 7, algoritams,1.3)
ver,klase=alg.predict(X)


















pom_x=X.iloc[:-45,:]
pom_y=y.iloc[:-45]
test_x=X.iloc[-45:,:]
#alg.learn(X, y, 7, algoritams)
alg=Ensable()
alg.learn(pom_x, pom_y, 7, algoritams)

#ver,klase=alg.predict(X)
ver,klase=alg.predict(test_x)
ver
