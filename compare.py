import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from datetime import datetime, timedelta
from streamlit_date_picker import date_range_picker, date_picker, PickerType
# matplotlib.use('TkAgg')

def plot_anomalies(ticker, anom_num, scaled_data,stdt,eddt, model1='DBSCAN',model2='IsolationForest'):
    '''
    model = 'statistical', 'DBSCAN', 'IsolationForest', 'OCSVM', 'Autoencoder'
    '''
    # Filter the data for the specified ticker
    data_tic_all = scaled_data[scaled_data['tic'] == ticker].copy()
    data_tic = data_tic_all[(data_tic_all['date'] >= stdt) & (data_tic_all['date']<= eddt)].copy()
    nds=len(data_tic)

    # calculate return data
    data_tic['return'] = data_tic['close'].pct_change(fill_method=None)
    data_tic['log_volume'] = np.log(data_tic['volume']+1)

    # Filter for anomalies with highest 'Anomaly Probability' for the desired number
    anomalies1 = data_tic.sort_values(by=f'{model1}_Anomaly_Probability', ascending=False).head(min(nds,anom_num))
    nps1=len(anomalies1)
    anomalies2 = data_tic.sort_values(by=f'{model2}_Anomaly_Probability', ascending=False).head(min(nds,anom_num))
    nps2=len(anomalies2)

    # Plotting
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(16,16))

    # Plot the close price and volume
    ax[0].plot(data_tic['date'], data_tic['close'], label='Close Price')
    ax[0].set(title = f'{ticker} Price Anomalies')

    ax[1].plot(data_tic['date'], data_tic['return'], label='Return')
    ax[1].set(title = f'{ticker} Return Anomalies')
    
    ax[2].fill_between(data_tic['date'], 0, data_tic['log_volume'], label='log Volume', alpha=0.8)
    ax[2].set(title=f'{ticker} Volume Anomalies')
    

    # Mark anomalies for model 1
    ax[0].scatter(anomalies1['date'], anomalies1['close'], color='red', label=f'{model1} Anomaly', marker='^')
    ax[1].scatter(anomalies1['date'], anomalies1['return'], color='red', label=f'{model1} Anomaly', marker='^')
    ax[2].bar(anomalies1['date'], anomalies1['log_volume'], color='red', label=f'{model1} Anomaly', width=1)
    
    # Mark anomalies for model 2
    ax[0].scatter(anomalies2['date'], anomalies2['close'], color='green', label=f'{model2} Anomaly', marker='^')
    ax[1].scatter(anomalies2['date'], anomalies2['return'], color='green', label=f'{model2} Anomaly', marker='^')
    ax[2].bar(anomalies2['date'], anomalies2['log_volume'], color='green', label=f'{model2} Anomaly', width=1)

    
    # show legend and xlabel
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.xlabel('Date')

    return fig,nps1,nps2,nds

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

data = pd.read_csv("df_final_prob.csv", parse_dates=['date'])

word_match = { # show : colname
    'DBSCAN': 'DBSCAN',
    'Isolation Forest': 'IsolationForest',
    'One-class SVM' : 'OCSVM',
    #'Statistical Model' : 'stat',
    #'Autoencoder': 'Autoencoder',
}

# data = data[['date', 'tic', 'close', 'volume',
#        'DBSCAN_Anomaly_Probability', 'IsolationForest_Anomaly_Probability', 'OCSVM_Anomaly_Probability']]

ticker_list = data['tic'].unique()
model_list = list(word_match.keys())

# ticker = 'ARL'
# model = 'Statistical Model'

ticker = st.selectbox(
    "Select a ticker",
    ticker_list,
    placeholder='Select...'
)

model1 = st.selectbox(
    "Select a model",
    model_list,
    placeholder='Select...'
)

model_list_copy=model_list[::-1]
model_list_copy.remove(model1)
model1 = word_match[model1]
model2 = st.selectbox(
    "Select another model",
    model_list_copy,
    placeholder='Select...'
)
model2 = word_match[model2]

anom_num=st.number_input(
    "Input the desired number of anomalies ", min_value=1, value=50, step=1, placeholder="Type an integer..."
)
st.write(f"The number of anomalies allowed is {anom_num}.")

default_start= datetime(2015, 10, 20)
default_end=datetime(2024, 8, 29)
dstart, dend = default_start,default_end
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


if ticker and anom_num and model1 and model2 and date_range_string:
    fig,nps1,nps2,nds = plot_anomalies(ticker, anom_num, data,dstart, dend, model1,model2)
    st.pyplot(fig)
    st.write(f"{nps1} anomalies detected by {model1}")
    st.write(f"and {nps2} anomalies detected by {model2}")
    st.write(f"within {nds} days.")
    # plt.show()
