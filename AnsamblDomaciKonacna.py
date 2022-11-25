#%%
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


#%%
class Ensable:
    
    def learn(self,X,y,ensemble_size,algoritams,learning_rate):
        X = pd.get_dummies(X)
        n,m = X.shape
        alfas = pd.Series(np.array([1/n]*n), index=X.index)
        self.ensemble = []
        self.weights = np.zeros(ensemble_size)        

        for t in range(ensemble_size):        	        	
        	alg=deepcopy(algoritams[random.randint(0,len(algoritams)-1)])
        	model = alg.fit(X,y, sample_weight=alfas)
        	predictions = model.predict(X)        	
        	error = (predictions!=y).astype(int)
        	weighted_error = (error*alfas).sum() 
        	w = 1/2 * math.log((1-weighted_error)/weighted_error) #*learning_rate  # preracunavanje tezina modela
        	self.ensemble.append(model)
        	self.weights[t] = w
        	factor = np.exp(-w*predictions*y*learning_rate ) 
        	alfas = alfas * factor   #preracunavanje novih tezina instanci, po formuli        	
        	z = alfas.sum()   #normalizacija
        	alfas = alfas/z
            

#bolje bi bilo da vecina delova iz predicta ide u learn

    def predict(self,data,y):   
        X=pd.get_dummies(data)
        predictions = pd.DataFrame([model.predict(X) for model in self.ensemble]).T        
        predictions['ensemble'] = np.sign(predictions.dot(self.weights))
        probs=pd.DataFrame([model.predict_proba(X).max(axis=1) for model in self.ensemble]).T        
        print((predictions.add(y, axis=0).abs()/2).mean())
        pom=(predictions.add(predictions['ensemble'], axis=0).abs()/2).drop('ensemble',axis=1) #paznja! (y moze umesto)
        sum_weights=sum(self.weights)
        verovatnoce_tacnih=pom*probs
        
        #racunanje po rezultatima modela
        #pom.apply(lambda x:x.dot(self.weights)/sum_weights,axis=1)
        return verovatnoce_tacnih.apply(lambda x:x.dot(self.weights)/sum_weights,axis=1),predictions['ensemble'],predictions
    

#%%
algoritams=[GaussianNB(),DecisionTreeClassifier(max_depth=2)]
alg=Ensable()
alg.learn(X, y, 7, algoritams,1)
ver,klase,predictions=alg.predict(X,y)



#%%
