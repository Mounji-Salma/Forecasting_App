# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#streamlit run ipykernel_launcher 
from math import ceil
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import pmdarima as pm
import random
random.seed(7)
np.random.seed(7)
import warnings
warnings.filterwarnings('ignore')
import joblib
import streamlit as st
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock



# %%
def remove_axes(n_used_axes, all_axs):
    if n_used_axes%2 :
        all_axs = all_axs.flatten()
        size = len(all_axs)
        for useless_ax in all_axs.flatten()[-(size - n_used_axes):]:
            useless_ax.axis('off')


# %%
def auto_arima_for_df(df, trace: bool):
    """find best arima model for each series in a data frame, returns dict of models"""
    models = {}
    for col in df.columns:
        if trace : print('\n',col, ':')
        models[col] = pm.auto_arima(df[col], start_p=1, start_q=1,
                              test='adf',       # use adftest to find optimal 'd'
                              max_p=3, max_q=3, # maximum p and q
                              m=12,              # frequency of series
        #                       d=None,           # let model determine 'd'
        #                       seasonal=True,   # No Seasonality
        #                       start_P=0, 
        #                       D=0, 
                              trace=trace,
        #                       error_action='ignore',  
        #                       suppress_warnings=True, 
                              stepwise=True)
    return models

# %%
@st.cache
def import_data(path):
    try:
        sales = pd.read_csv(path, index_col=0, parse_dates=True)
        print("data imported")
    except FileNotFoundError:
        st.write("NotFoundError")
        raise
    return sales

sales = import_data('70prod_data.csv')

# %%

def hash_joblib_reference(file_reference):
    return True

@st.cache(hash_funcs={dict: hash_joblib_reference})
def import_models(path):
    try:
        models = joblib.load(path)
        print(type(models))
        print("models imported")
        return models
    except FileNotFoundError:
        st.write("NotFoundError")
        raise

models = import_models("joblib_ARIMA_sales_Models.joblib")


# %%
def empty_df_like(df):
    return pd.DataFrame(columns=df.columns, index= df.index)
def empty_df_to_date(df, n_periods):
    return pd.DataFrame(columns=df.columns,index= pd.date_range(df.index[-1], freq="M", periods=n_periods).shift(1, freq='M'))


# %%
def arima_dynamic_out_of_sample_forecast(models, train_df, n_periods) :
    """enerate future dynamic forecast to the given periods"""
    fc_df = empty_df_to_date(train_df, n_periods)
    lower_df = empty_df_to_date(train_df, n_periods)
    upper_df = empty_df_to_date(train_df, n_periods)
    
    for  col in train_df.columns:
        fitted_dynamic = models[col].fit(train_df[col])
        fc, confint = fitted_dynamic.predict(n_periods= n_periods, return_conf_int=True)
        
        fc_df[col] = fc
        lower_df[col] = confint[:, 0]
        upper_df[col] = confint[:, 1]
    return {'fc_df' : fc_df, 'lower_df' : lower_df, 'upper_df' : upper_df}


# %%
def plot_forecasts(forecast_data, df):
    n_series = len(df.columns)
    fig_n_lines = ceil(n_series/2)
    # fig, axs = plt.subplots(fig_n_lines, 2 if n_series>1 else 1, figsize=(20, fig_n_lines*4))
    with _lock:
        if n_series>1 : 
            fig, axs = plt.subplots(fig_n_lines, 2, figsize=(20, fig_n_lines*4))
            remove_axes(n_series, axs)
            axs = axs.flatten()
        else :
            fig, axs = plt.subplots(fig_n_lines, 1, figsize=(15, 7))
            axs = [axs]

        for i, item in enumerate(df.columns):
            axs[i].plot(pd.concat([df[item], forecast_data['fc_df'][item]]), label="actual")
            axs[i].plot(forecast_data['fc_df'][item], color='darkgreen', label="forecast")
            axs[i].set_title(f"{item} forecast")

            lower = forecast_data['lower_df'][item]
            upper = forecast_data['upper_df'][item]
            axs[i].plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)
            axs[i].plot(upper, "r--", alpha=0.5)
            
            axs[i].axvspan(lower.index[0], lower.index[-1],  color=sns.xkcd_rgb['grey'], alpha=0.2)

            axs[i].legend(loc="best")

        st.pyplot(fig)



# %%
# st.set_page_config(page_title='Sales Forcasting App', layout='wide')

st.write("# Sales Forcasting App")
# Sidebar - Collects user input features into dataframe
with st.sidebar:
    st.write('## Select products')
    cols = list(sales.columns)
    # container = st.beta_container()
    all_p = st.checkbox("Select all products")

    if all_p:
        selected_products = cols
    else:
        selected_products =  st.multiselect("Select one or more products :", \
            cols)

    st.write('## Select forecast horizon')      

    dates = list(pd.Series(pd.date_range(\
        start='2018-02-28', end='2019-09-30', freq='M')).astype(str))
    selected_date = st.selectbox('Select first month :', dates[1:], \
        format_func=lambda date: date[:-3])
    last_train_date = dates[dates.index(selected_date) - 1]
    train_df=sales.loc[:last_train_date, selected_products]
    n_periods = st.sidebar.slider('Select number of months :', 1, 24, 6, 1)

import base64  
def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="Forecast.csv">Download csv file</a>'
    return href

import itertools

if st.sidebar.button('ok') : 
    res = arima_dynamic_out_of_sample_forecast(models, train_df, n_periods)
    pred_df = pd.DataFrame(columns=list(itertools.chain.from_iterable([[col+'_Forecast', col+'_LowerB', col+'_UpperB'] for col in train_df.columns])), index = res['fc_df'].index)
    for col in train_df.columns:
        pred_df[col+'_Forecast'] = res['fc_df'][col]
        pred_df[col+'_LowerB'] = res['lower_df'][col]
        pred_df[col+'_UpperB'] = res['upper_df'][col]
    pred_df.index = pred_df.index.astype(str)
    st.dataframe(pred_df)
    st.markdown(get_table_download_link(pred_df), unsafe_allow_html=True)
    plot_forecasts(res, train_df)

# %%


# %%

