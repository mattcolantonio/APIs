#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 12:26:59 2023

@author: matthewcolantonio
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import fredapi as fd 
 
# set up the connection with fred database via apl key
fred=fd.Fred(api_key="35b8be6037f7722d989a3e1d2040bfd2")

# extract data from FRED. This forecast is for PPI for Oil and Gas Extraction
data = fred.search('Producer Price Index by Industry: Oil and Gas Extraction')
data.head(10)

data['title'][0]

ppi = fred.get_series('PCU21112111')
ppi.name = 'values'
ppi

df = pd.DataFrame(ppi).reset_index()
df

# starting from 1986 
df2=df[df['index']>'1985-12-01']

fig1 = px.line(df2, x='index', y='values', title = 'PPI Oil and Gas Extraction, 1986-2023')
fig1.show()

# Efficient (Facebook) Prophet Model conditions
# 1 Univariate data
# 2 predetermined to forecast daily data 
# 3 Can only read two column names, date:ds and variable:y

df2 = df2.rename(columns={'index': 'ds', 'values': 'y'})
df2



from prophet import Prophet

ml=Prophet()
ml.fit(df2)

future=ml.make_future_dataframe(periods=10, freq='MS')
future.tail()

result = ml.predict(future)
result
result[{'ds', 'yhat', 'yhat_lower', 'yhat_upper'}]

fig2=ml.plot(result)

fig3=ml.plot_components(result)

# should also consider cross-validation in time series modelling
import cross_validation.performance_metrics

cv_results=cross_validation(model=ml, initial=pd.to_timedelta(30*20, unit='D'), period=pd.to_timedelta(30*5, unit='D'), horizon=pd.to_timedelta(30*12, unit='D'))

import plot_cross_validation_metric

fig4=plot_cross_validation_metric(cv_results, metric=rmse
                                  
                                  )


