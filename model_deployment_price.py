#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.decomposition import PCA
import sys
import os


# In[ ]:


def predict_price(year, mileage, state, make, model):
    # 
    reg = joblib.load(os.path.dirname(__file__) + '/price_reg.pkl')
    year_mil = pd.DataFrame(data = [[year,mileage]], columns = ['Year', 'Mileage']).astype(int)
    categoricas = pd.DataFrame(data = [[state, make, model]],columns = ['State', 'Make', 'Model']).astype(str)
    data_input = pd.merge(year_mil,categoricas, left_index = True, right_index=True)
    #print(data_input)
    
    data = pd.read_csv('dataTrain_carListings.csv')
    
    data = data.append(data_input)
    
    X_2 = pd.get_dummies(data[['State', 'Make', 'Model']])
    pca = PCA(n_components = 100).fit(X_2)
    #print(pca)
    
    componentes = []
    
    for i in range(0,99):
        comp = np.dot(X_2[-1:], pca.components_[i])
        componentes.append(comp)
    pcadf = pd.DataFrame(np.vstack((componentes)).T)
    
    X = pd.merge(year_mil, pcadf, left_index = True, right_index=True)
    #print(X)
    
    #Make prediction
    pred_precio = reg.predict(X)
    
    return pred_precio

if __name__ == "__main__":
    
    pred_precio = predict_price(year, mileage, state, make, model)
    print(year, mileage, state, make, model)
    print("Precio pronosticado", pred_precio)

