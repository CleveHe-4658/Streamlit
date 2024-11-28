import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from datetime import datetime, timedelta
from streamlit_date_picker import date_range_picker, date_picker, PickerType
# matplotlib.use('TkAgg')

def plot_anomalies(ticker, scaled_data,stdt,eddt, model='DBSCAN'):
    '''
    model = 'statistical', 'DBSCAN', 'IsolationForest', 'OCSVM', 'Autoencoder'
    '''
    # Filter the data for the specified ticker
    data_tic_all = scaled_data[scaled_data['tic'] == ticker].copy()
    data_tic = data_tic_all[(data_tic_all['date'] >= stdt) & (data_tic_all['date']<= eddt)].copy()

    # calculate return data
    data_tic['return'] = data_tic['close'].pct_change(fill_method=None)
    data_tic['log_volume'] = np.log(data_tic['volume']+1)

    # Filter for anomalies where 'Anomaly' equals 1
    anomalies = data_tic[data_tic[f'{model}_Anomaly'] == 1]

    # Plotting
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(16,16))

    # Plot the close price and volume
    ax[0].plot(data_tic['date'], data_tic['close'], label='Close Price')
    ax[0].set(title = f'{ticker} Price Anomalies')

    ax[1].plot(data_tic['date'], data_tic['return'], label='Return')
    ax[1].set(title = f'{ticker} Return Anomalies')
    
    ax[2].fill_between(data_tic['date'], 0, data_tic['log_volume'], label='log Volume', alpha=0.8)
    ax[2].set(title=f'{ticker} Volume Anomalies')
    

    # Mark anomalies
    ax[0].scatter(anomalies['date'], anomalies['close'], color='red', label=f'{model} Anomaly', marker='^')
    ax[1].scatter(anomalies['date'], anomalies['return'], color='red', label=f'{model} Anomaly', marker='^')
    ax[2].bar(anomalies['date'], anomalies['log_volume'], color='red', label=f'{model} Anomaly', width=1)
    
    # show legend and xlabel
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.xlabel('Date')

    return fig

st.title("Anomaly Visualization")
st.write('Bloomberg capstone group Bravo:')
st.write("Xinran Cheng, Zhaoyang Hong, Qi Wu, Haoran Yang, Cleve He")

st.write("""\n
This is a visualization of the detected anomalies by our models. 
Available stock universe is first 17 of the 200 least liquid stocks in Russell 2000. 
Available models includes statistical, DBSCAN, Isolation forest, One-class SVM, Autoencoder, and LSTM.\n
To view the labelled anomalies, choose one ticker and one model type from the selection bar.
Time horizon ranges from 2015-10-20 to 2024-08-29 and can be adjusted by a range slider. 
""")

data = pd.read_csv("df_final_merged_renewed.csv", parse_dates=['date'])

word_match = { # show : colname
    'DBSCAN': 'DBSCAN',
    'Isolation Forest': 'IsolationForest',
    'One-class SVM' : 'OCSVM',
    'Statistical Model' : 'stat',
    'Autoencoder': 'Autoencoder',
}

# data = data[['date', 'tic', 'close', 'volume',
#        'DBSCAN_Anomaly', 'IsolationForest_Anomaly', 'OCSVM_Anomaly',
#        'Autoencoder_Anomaly', 'stat_Anomaly']]

ticker_list = data['tic'].unique()
model_list = list(word_match.keys())

# ticker = 'ARL'
# model = 'Statistical Model'

ticker = st.selectbox(
    "Select a ticker",
    ticker_list,
    placeholder='Select...'
)

model = st.selectbox(
    "Select a model",
    model_list,
    placeholder='Select...'
)
model = word_match[model]

default_start= datetime(2015, 10, 20)
default_end=datetime(2024, 8, 29)
available_datas = []
date_range = default_start
while date_range <= default_end:
    available_datas.append(date_range)
    date_range += timedelta(days=1)

date_range_string = date_range_picker(picker_type=PickerType.date,
                                      start=default_start, end=default_end,
                                      available_dates=available_datas,
                                      key='available_date_range_picker',)
if date_range_string:
    dstart, dend = date_range_string
    st.write(f"Date Range Picker [{dstart}, {dend}]")

if ticker and model and date_range_string:
    fig = plot_anomalies(ticker, data,stdt,eddt, model)
    st.pyplot(fig)
    # plt.show()
