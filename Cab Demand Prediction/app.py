import streamlit as st
import pandas as pd
from datetime import datetime,date
import time as t
import datetime

data = pd.read_csv("data_with_all_preds.csv")

st.markdown("<h2 style='text-align: center; color: black;'>CAB DEMAND PREDICTION</h2>",unsafe_allow_html=True)

col1,col2,col3 = st.columns(3)
col4,col5,col6 = st.columns(3)

col1.markdown("<h4 style='text-align: center; color: black;'>Region-ID</h4>",unsafe_allow_html=True)
region = col4.selectbox('',(str(i) for i in range(30)))

col2.markdown("<h4 style='text-align: center; color: black;'>Date</h4>",unsafe_allow_html=True)
date = col5.date_input('',key='date',value=date(2016,6,1),min_value=date(2016,6,1),max_value=date(2016,6,30))

col3.markdown("<h4 style='text-align: center; color: black;'>Time</h4>",unsafe_allow_html=True)
time = col6.time_input('',key='time',value=None)


st.markdown("<h4 style='text-align: center; color: black;'>Choose Technique</h4>",unsafe_allow_html=True)
col8,col9,col10 = st.columns([4,8,1])
tech = col9.radio("",('SMA (Simple Moving Average)', 'WMA (Weighted Moving Average)', 'EWMA (Exponential Weighted Moving Average)'))

colx,coly,colz = st.columns([6,8,1])
result = coly.button(label='Predict Pickups',key='button')
if result:
    time_stamp_from_1970 = t.mktime(datetime.datetime.strptime(str(date.year)+'-'+str(date.month)+'-'+str(date.day)+" "+str(time.hour)+':'+str                                    (time.minute)+':'+str(time.second), "%Y-%m-%d %H:%M:%S").timetuple())
    first_jun_time_stamp = t.mktime(datetime.datetime.strptime("2016-06-01 00:00:00", "%Y-%m-%d %H:%M:%S").timetuple())
    time_bin = (time_stamp_from_1970-first_jun_time_stamp)//600
    if tech == 'SMA (Simple Moving Average)':
        pred = data.query('Region == {} and time_bins == {}'.format(region,time_bin))['sma_predictions']
        st.markdown("<h4 style='text-align: center; color: black;'>Using SMA technique, the pick-ups that could be demanded for the given region and time                    stamp is {} with percentage Error of 17.92.</h4>".format(round(pred.values[0])),unsafe_allow_html=True)
    elif tech == 'WMA (Weighted Moving Average)':
        pred = data.query('Region == {} and time_bins == {}'.format(region,time_bin))['wma_predictions']
        st.markdown("<h4 style='text-align: center; color: black;'>Using WMA technique, the pick-ups that could be demanded for the given region and time                     stamp is {} with percentage Error of 12.90.</h4>".format(round(pred.values[0])),unsafe_allow_html=True)
    elif tech == 'EWMA (Exponential Weighted Moving Average)':
        pred = data.query('Region == {} and time_bins == {}'.format(region,time_bin))['ewma_predictions']
        st.markdown("<h4 style='text-align: center; color: black;'>Using EWMA technique, the pick-ups that could be demanded for the given region and time                     stamp is {} with Percentage Error of 15.96.</h4>".format(round(pred.values[0])),unsafe_allow_html=True)

    #st.write(pred.values[0])
    #st.markdown("<h3 style='text-align: center; color: black;'>Region-ID</h3>",unsafe_allow_html=True)











