import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
# matplotlib.use('TkAgg')

def plot_anomalies(ticker, scaled_data, model='DBSCAN'):
    '''
    model = 'statistical', 'DBSCAN', 'IsolationForest', 'OCSVM', 'Autoencoder'
    '''
    # Filter the data for the specified ticker
    data_tic = scaled_data[scaled_data['tic'] == ticker].copy()

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
Available stock universe is the 200 least liquid stocks in Russell 2000. Time horizon ranges from Oct. 2014 to Oct. 2024. Available models includes statistical, DBSCAN, Isolation forest, One-class SVM, Autoencoder, and LSTM.\n
To view the labelled anomalies, choose one ticker and one model type from the selection bar.
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

if ticker and model:
    fig = plot_anomalies(ticker, data, model)
    st.pyplot(fig)
    # plt.show()