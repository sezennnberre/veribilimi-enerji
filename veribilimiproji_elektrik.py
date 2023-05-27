# -*- coding: utf-8 -*-
"""
Created on Sat May 27 13:52:10 2023

@author: HP
"""

import numpy as np 
import pandas as pd


hava = pd.read_csv("weather_features.csv")
enerji = pd.read_csv("energy_dataset.csv")

hava_durumu = hava.iloc[:,2:14].values
hava_durumu = pd.DataFrame(data = hava_durumu,columns=["temp","temp_min","temp_max","pressure","humidity",
                                                       "wind_speed","wind_deg","rain_1h","rain_3h","snow_3h",
                                                       "clouds_all","weather_id"])
enerji_üretimi = enerji.iloc[:,1:25].values
enerji_üretimi = pd.DataFrame(data = enerji_üretimi,columns=["generation biomass","generation fossil coal/lignite","generation fossil coal-derived gas",
                                                             "generation fossil gas","generation hard coal","generation oil","generation oil shale",
                                                             "generation fossil peat","generation geothermal","generation hydro1",
                                                             "generation hydro2","generation hydro3","generation hydro4", 
                                                             "generation marine","generation nuclerar","generation other",
                                                             "generation other renewable","generation solar","generation waste",
                                                             "generation wind offshore","generation wind onshore","forecast solar day ahead",
                                                             "forecast wind offshore day ahead","forecast wind onshore day ahead"])

enerji_üretimi.drop(["generation hydro1","forecast wind offshore day ahead"],axis = 1,inplace=True)
enerji_üretimi.dropna(inplace=True)

değerler = pd.concat([hava_durumu,enerji_üretimi],axis=1)
değerler.dropna(inplace=True)
değerler.drop([35052,35053,35054,35055,35056,35057,35058,35059,35060,35061,35062,35063],axis =0,inplace=True)

tahmin1 = enerji.iloc[:,26:27].values
tahmin1 = pd.DataFrame(data=tahmin1,columns=["total load"])
tahmin2 = enerji.iloc[:,28:].values
tahmin2 = pd.DataFrame(data=tahmin2,columns=["price"])
tahminverisi = pd.concat([tahmin1,tahmin2],axis=1)
tahminverisi.dropna(inplace=True)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(değerler,tahminverisi,train_size=0.33,random_state=0)


#çoklu doğrusal regression kullanıldı.
from sklearn.linear_model import LinearRegression
lr  = LinearRegression()
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)

#çoklu doğrusal regression için r^2 kullanıldı.
from sklearn.metrics import r2_score
r2 = r2_score(y_train, lr.predict(x_train))
print("Doğruluk oranı: ", r2*100)

from sklearn.metrics import accuracy_score


#decision tree kullanıldı
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state=0)
dtr.fit(x_train,y_train)

y_pred_dtr = dtr.predict(x_test)

#decision tree için r^2 kullanıldı.
from sklearn.metrics import r2_score
r2 = r2_score(y_train, dtr.predict(x_train))
print("Doğruluk oranı: ", r2*100)


#random forest regression kullanıldı.
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=10, n_jobs=-1)
rfr.fit(x_train,y_train)

y_pred_rfr = rfr.predict(x_test)

# random forest regressor için r^2 kullanıldı.
from sklearn.metrics import r2_score
r2 = r2_score(y_train, rfr.predict(x_train))
print("Doğruluk oranı: ", r2*100)














