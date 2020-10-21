#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup
import requests
import lxml
import pandas as pd
import numpy as np
from  tqdm import tqdm
import time
from datetime import datetime
import io
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors
from matplotlib.font_manager import FontProperties
import math
import plotly.graph_objects as go
from sklearn.metrics import r2_score
import scipy.optimize
import fbprophet
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from fbprophet import Prophet
#from scipy.optimize import curve_fit
# ignore InsecureRequestWarning
import certifi
import urllib3
urllib3.disable_warnings()


# # FUNCTIONS

# In[2]:


def rearrange_df(df):
    list_df=[]
    continent_list=df['continent'].unique()
    for i in range(len(continent_list)):
        list_df.append([])
    iso_code_list=list(df['iso_code'].drop_duplicates())
    for i in range(len(iso_code_list)):
        temp=df[df['iso_code']==iso_code_list[i]].reset_index(drop=True)
        for j in range(len(continent_list)):
            if temp['continent'][0]==continent_list[j]:
                list_df[j].append(temp)
    return list_df


# In[3]:


def plot_graph(df,title,x_title,y_title):
    import plotly.graph_objects as go
    column=df.columns
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[column[0]], y=df[column[1]],mode='lines + markers'))
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title
    )
    fig.write_html("normal.html")
    fig.show()


# In[4]:


#return list
def culumlative(df,column):
    result=[]
    temp=list(df[column])
    for i in range(len(temp)):
        result.append(sum(temp[0:i]))
    return result


# In[5]:


def logistic_growth_plot(df,y_column,function):
    #import scipy.optimize
    y=new_df[y_column]
    x=np.arange(len(y))
    bounds=(0,[6000.,25.,10708981.])
    params,_=scipy.optimize.curve_fit(function,x,y,bounds=bounds)
    #params,_=curve_fit(function,x,y,bounds=bounds)
    print('params:')
    print(params)
    plt.plot(x,y,'.',label='observations')
    y_fit=function(x,*params)
    print('r2_score: "%6.2f"'%(r2_score(y,y_fit)))
    print ('mean squared error: "%6.2f"' %(mean_squared_error(y,y_fit)))
    plt.plot(x,y_fit,label='fitted curved')
    plt.legend()
    plt.savefig('logistic model.png')
    plt.show()
    return (pd.concat([pd.DataFrame(x,columns=['date']),pd.DataFrame(list(new_df[y_column]),columns=['original']),pd.DataFrame(y_fit,columns=['predicted'])],axis=1)),params


# In[6]:


def predict_prophet_logistic(data_train,data_test,cap,plot):
    model=Prophet(growth='logistic')
    model.fit(data_train)
    future_dates=model.make_future_dataframe(periods=365)
    future_dates['cap']=cap
    prediction=model.predict(future_dates)
    if plot==1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data_train.index.tolist(), y=data_train['y'].to_list(),mode='lines + markers',name='training data'))
        fig.add_trace(go.Scatter(x=data_test.index.tolist(), y=data_test['y'].to_list(),mode='lines + markers',name='test data'))
        fig.update_layout(
            title='Cumulative number of confirmed COVID 19 cases in Czech Republic---training data vs test data',
            xaxis_title='Date',
            yaxis_title='Number of cases'
        )
        fig.write_html("data_set_logistic.html")
        fig.show()
        model.plot_components(prediction)
    return(prediction)


# In[7]:


def predict_prophet_regular(data_train,data_test,plot):
    model=Prophet()
    model.add_seasonality(name='monthly',period=30.5,fourier_order=5)
    model.fit(data_train)
    future_dates=model.make_future_dataframe(periods=365)
    forecast=model.predict(future_dates)
    #model.plot_components(forecast)
    if plot==1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data_train.index.tolist(), y=data_train['y'].to_list(),mode='lines + markers',name='training data'))
        fig.add_trace(go.Scatter(x=data_test.index.tolist(), y=data_test['y'].to_list(),mode='lines + markers',name='test data'))
        fig.update_layout(
            title='Cumulative number of confirmed COVID 19 cases in Czech Republic---training data vs test data',
            xaxis_title='Date',
            yaxis_title='Number of cases'
        )
        fig.write_html("data_set_regular.html")
        fig.show()
        model.plot_components(forecast)
    return(forecast)


# In[8]:


#return training set data and test data 
#cut off: 1:through date, 0:through %
def split_data(or_df,x_col_name,y_col_name,cut_off,percent,date):
    temp_df=or_df[[x_col_name,y_col_name]]
    temp_df=temp_df.rename(columns={x_col_name:'ds',y_col_name:'y'})
    if cut_off==1:
        temp_df=temp_df.set_index('ds')
        train_df=temp_df.loc[temp_df.index<=date].copy()
        test_df=temp_df.loc[temp_df.index>date].copy()
    else:
        split_num=temp_df.shape[0]*percent
        if type(split_num*percent)!=int:
            split_num+=1
        train_df=temp_df.loc[temp_df.index<=split_num].copy()
        test_df=temp_df.loc[temp_df.index>split_num].copy()
        #train_df=train_df.set_index('ds')
        #test_df=test_df.set_index('ds')
    return train_df,test_df


# In[9]:


# plotting the time series model using plotly
def plot_time_series(forecast,original_df,title,x_name,y_name,data_train, data_test):
    fig = go.Figure()
    yhat = go.Scatter(
      x = forecast['ds'],
      y = forecast['yhat'],
      mode = 'lines',
      marker = {
        'color': '#3bbed7'
      },
      line = {
        'width': 3
      },
      name = 'Forecast',
    )
    yhat_lower = go.Scatter(
      x = forecast['ds'],
      y = forecast['yhat_lower'],
      marker = {
        'color': 'rgba(0,0,0,0)'
      },
      showlegend = False,
      hoverinfo = 'none',
    )
    yhat_upper = go.Scatter(
      x = forecast['ds'],
      y = forecast['yhat_upper'],
      fill='tonexty',
      fillcolor = 'rgb(233, 164, 73)',
      name = 'Confidence',
      hoverinfo = 'none',
      mode = 'none'
    )
    """
    actual = go.Scatter(
      x = original_df['date'],
      y = original_df['new_cases'],
      mode = 'markers',
      marker = {
        'color': '#fffaef',
        'size': 4,
        'line': {
          'color': '#000000',
          'width': .75
        }
      },
      name = 'Actual'
    )
    """
    training = go.Scatter(
      x = data_train['ds'],
      y = data_train['y'],
      mode = 'markers',
      marker = {
        'color': '#fffaef',
        'size': 4,
        'line': {
          'color': '#000000',
          'width': .75
        }
      },
      name = 'training data',
    )
    testing = go.Scatter(
      x = data_test['ds'],
      y = data_test['y'],
      mode = 'markers',
      marker = {
        'color': '#588763',
        'size': 4,
        'line': {
          'color': '#000000',
          'width': .75
        }
      },
      name = 'testing data'
    )
    fig.add_trace(yhat_lower)
    fig.add_trace(yhat_upper)
    fig.add_trace(yhat)
    #fig.add_trace(actual)
    fig.add_trace(training)
    fig.add_trace(testing)
    fig.update_layout(
            title=title,
            xaxis_title=x_name,
            yaxis_title=y_name
        )
    fig.write_html("model.html")
    fig.show()


# In[10]:


def measure_accuracy(original,predict):
    temp=pd.concat([original,predict['yhat']],axis=1)
    temp=temp.dropna()
    r_2_score=r2_score(temp.new_cases,temp.yhat)
    mean_square_error=mean_squared_error(temp.new_cases,temp.yhat)
    print('r2_score: %.6f' %(r_2_score))
    print('mean_squared_error: %.6f' %(mean_square_error))
    return ([r_2_score,mean_square_error])


# In[11]:


#cut_off:60%-90% for training,cut_off_start=0.6,cut_off_end=0.9
#model:{logistic=0,regular=1}
#comparsion_method:{r2=0,mean_square=1}
#cap: for logistic model only, else=0
def cross_validate(original_df,cut_off_start,cut_off_end,sequence,model,cap,comparsion_method):
    r_2_error_list=[]
    mean_square_list=[]
    prediction=[]
    percent=[]
    for i in np.arange(cut_off_start,cut_off_end+sequence, sequence):
        print(i)
        percent.append(i)
        data_train,data_test=split_data(ts_df,'date','new_cases',0,i,'')
        if model==0:
            data_train['cap']=cap
            predict=predict_prophet_logistic(data_train,data_test,cap,0)
            result=measure_accuracy(original_df,predict)
        else:
            predict=predict_prophet_regular(data_train,data_test,0)
            result=measure_accuracy(original_df,predict)
        r_2_error_list.append(result[0])
        mean_square_list.append(result[1])
        prediction.append(predict)

    if comparsion_method==0:
        position=r_2_error_list.index(max(r_2_error_list))
    else:
        position=mean_square_list.index(min(mean_square_list))
    return (percent[position],prediction[position])


# In[12]:


def R_t_calculation(df):
    temp=df.copy()
    result=[]
    for i in range(df.shape[0]):
        if (df['new_deaths'][i]+df['new_recover'][i])!=0:
            rt=(df['new_cases'][i]/(df['new_deaths'][i]+df['new_recover'][i]))+1
        else:
            rt=(df['new_cases'][i]/(df['new_deaths'][i]+df['new_recover'][i]+0.05))+1
        result.append(rt)
    temp['R_t']=result
    return (temp,temp[['date','R_t']])


# # MAIN SCRIPT

# STEP 1: READ IN DATA

# In[13]:


url="https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
s=requests.get(url).content
df=pd.read_csv(io.StringIO(s.decode('utf-8')))


# In[14]:


# scope down to czech republic


# In[15]:


scope=df[df['location']=='Czech Republic'].reset_index(drop=True)


# STEP 2: LOGISTIC GROWTH MODEL

# In[16]:


#cumulative function


# In[17]:


ts_df_cumul=scope[['date','new_cases']]
ts_df_cumul['new_cases']=culumlative(ts_df_cumul,'new_cases')


# In[18]:


plot_graph(ts_df_cumul,'Cumulative number of confirmed COVID 19 cases in Czech Republic','Date','Number of cases')


# In[19]:


# define parameter


# In[20]:


new_df=ts_df_cumul[ts_df_cumul['new_cases']!=0]


# In[21]:


b=1.1
c=10708981*0.8
y_0=ts_df_cumul[ts_df_cumul['new_cases']!=0].reset_index(drop=True)['new_cases'][0]
#a=
a=(c/y_0)-1


# In[22]:


def logistic_growth(x,a=a,b=b,c=c):
    return c/(1+a*scipy.special.expit(-b*x))


# In[23]:


#construction of logistic growth model


# In[24]:


ts_data,para=logistic_growth_plot(new_df,'new_cases',logistic_growth)


# STEP 3: TIME SERIES MODEL BASED ON LOGISTIC GROWTH FUNCTION

# In[25]:


#single trial


# In[26]:


ts_df=new_df.reset_index(drop=True)


# In[27]:


data_test


# In[28]:


data_train,data_test=split_data(ts_df,'date','new_cases',0,0.95,'')
data_train['cap']=para[-1]
predict=predict_prophet_logistic(data_train,data_test,para[-1],1)
plot_time_series(predict,ts_df,'Time Series Logistic Growth Model','date','infected case',data_train,data_test)
measure_accuracy(ts_df,predict)


# In[29]:


#cross validation


# In[30]:


percent,result_1=cross_validate(ts_df,0.8,0.95,0.03,0,para[-1],0)


# In[42]:


data_train,data_test=split_data(ts_df,'date','new_cases',0,percent,'')
data_train['cap']=para[-1]
predict=predict_prophet_logistic(data_train,data_test,para[-1],1)
plot_time_series(predict,ts_df,'Time Series Logistic Growth Model','date','infected case',data_train,data_test)
measure_accuracy(ts_df,predict)


# STEP 4: TIME SERIES MODEL WITH REGULAR LINEAR REGRESSION

# In[32]:


#single trial


# In[41]:


data_train,data_test=split_data(ts_df,'date','new_cases',0,0.95,'')
predict=predict_prophet_regular(data_train,data_test,1)
plot_time_series(predict,ts_df,'Time Series Linear regression Model','date','infected case',data_train,data_test)
measure_accuracy(ts_df,predict)


# In[34]:


#cross validation


# In[37]:


percent,result_1=cross_validate(ts_df,0.8,0.95,0.03,1,para[-1],0)


# In[38]:


data_train,data_test=split_data(ts_df,'date','new_cases',0,percent,'')
predict=predict_prophet_regular(data_train,data_test,1)
plot_time_series(predict,ts_df,'Time Series Linear regression Model','date','infected case',data_train,data_test)
measure_accuracy(ts_df,predict)


# # OPTIONAL PART FOR DISCUSSION

# STEP 5: PREDTICTION OF EFFECTIVE REPRODUCTION NUMBER

# In[8]:


#extraction of recovered case, infected case, death case


# In[132]:


recovery=[198,
239,
185,
167,
157,
32,
36,
823,
524,
469,
271,
154,
47,
38,
251,
286,
292,
236,
307,
44,
6,
260,
309,
210,
337,
364,
12,
14,
802,
284,
235,
413,
353,
65,
89,
726,
593,
318,
446,
481,
370,
527,
670,
726,
558,
673,
3358,
196,
332,
531,
960,
684,
743,
846,
259,
8858,
2487,
2152,
1979,
393,
2013,
1736,
71,
3320,
1701,
1000]


# In[138]:


recovery_2=[]
empty_num=len(scope['date'])-len(recovery)
for i in range(len(scope['date'])):
    if i<empty_num:
        recovery_2.append(0)
    else:
        recovery_2.append(recovery[i-empty_num])


# In[139]:


rt_df=pd.concat([scope[['date','new_deaths','new_cases']],pd.DataFrame(recovery_2,columns=['new_recover'])],axis=1).reset_index(drop=True)


# In[140]:


rt_df=rt_df[rt_df['date']>='2020-03-03'].reset_index(drop=True)


# In[144]:


2.2*rt_df['new_cases']/para[-1]


# In[107]:


rt_df_1,plot=R_t_calculation(rt_df)


# In[108]:


rt_df_1


# In[109]:


df=plot
x_title='Date'
y_title='Rt'
title='Rt in Czech Republic'
import plotly.graph_objects as go
column=df.columns
fig = go.Figure()
fig.add_trace(go.Scatter(x=df[column[0]], y=df[column[1]],mode='lines'))
fig.update_layout(
    title=title,
    xaxis_title=x_title,
    yaxis_title=y_title
)
fig.show()


# In[110]:


plot_graph(plot,'Rt in Czech Republic','Date','Rt')


# In[136]:


for j in rt_df.columns[1:]:
    rt_df[str(j+'new')]=culumlative(rt_df,j)


# In[137]:


rt_df


# In[131]:


rt_df['new_cases']=rt_df['new_cases']-rt_df['new_deaths']-rt_df['new_recover']


# In[128]:


rt_df


# In[129]:


rt_df_1,plot=R_t_calculation(rt_df)
plot_graph(plot,'Rt in Czech Republic','Date','Rt')


# In[112]:


culumlative(rt_df,'new_deaths')


# # REFERENCE CODE

# SIR MODEL

# In[13]:


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Total population, N.
N = 1000
# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 1, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
beta, gamma = 0.2, 1./10 
# A grid of time points (in days)
t = np.linspace(0, 160, 160)

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Initial conditions vector
y0 = S0, I0, R0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I/1000, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R/1000, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Time /days')
ax.set_ylabel('Number (1000s)')
ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()


# In[ ]:




