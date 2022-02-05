from keyword import kwlist
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from fbprophet import Prophet
import base64

from pytrends.request import TrendReq


## get google trends data from keyword list
@st.cache
def get_data(keyword):
    keyword = [keyword]
    pytrend = TrendReq()
    pytrend.build_payload(kw_list=keyword)
    df = pytrend.interest_over_time()
    try:
        df.drop(columns=['isPartial'], inplace=True)
        df.reset_index(inplace=True)
        df.columns = ["ds", "y"]
        return df
    except:
        pass

## make forecast for a new period
def make_pred(df, periods):
    periods = periods * 365
    prophet_basic = Prophet()
    prophet_basic.fit(df)
    future  = prophet_basic.make_future_dataframe(periods=periods)
    forecast = prophet_basic.predict(future)
    fig1 = prophet_basic.plot(forecast, xlabel="date", ylabel="trend", figsize=(10, 6))
    fig2 = prophet_basic.plot_components(forecast)
    forecast = forecast[['ds', 'yhat']]

    return forecast, fig1, fig2

## set streamlit page configuration
st.set_page_config(page_title="Trend Predictor", page_icon=":crystal_ball", layout='centered', initial_sidebar_state='auto')

## sidebar
st.sidebar.write("""
## Choose a keyword and a prediction period :dizzy:
""")
keyword = st.sidebar.text_input("Keyword", "facebook")
periods = st.sidebar.slider("Prediction time in years:", 1, 3, 5)
details = st.sidebar.checkbox("Show details",)

## main section
st.write("""
# Trend Predictor App :crystal_ball:
This app predicts the **Google Trend**
""")
## describing
expander_bar = st.expander("About", expanded=False)
expander_bar.markdown("""
* **Google Trends** 데이터를 가져와서 트렌드를 파악할 수 있습니다. 
* 여기에서는 시계열 분석 알고리즘을 이용하여 트렌드를 예측하도록 프로그램되어 있습니다.
* 왼쪽 사이드바에서 원하는 키워드와 예측할 기간(1~5년까지)만 정하면 됩니다.
---
* 주의사항 : **Google Trends** 가 영어를 기반으로 하기 때문에 간혹 한글 키워드를 입력하면 데이터가 안나올 수 있습니다. 
예를 들어 "의료데이터"를 입력하면 데이터가 추출이 안되지만 "healthcare data"로 하면 데이터가 도출됩니다.
""")


st.write("Evolution of interest:", keyword)
df = get_data(keyword)
if df is not None:
    forecast, fig1, fig2 = make_pred(df, periods)
    st.pyplot(fig1) 
    
    if details:
        st.write("### Details :mag_right:")
        st.pyplot(fig2)
        

else:
    st.write("데이터가 없습니다. 키워드를 다시 설정해 주세요!")

# Download dataframe
def filedownload(df):
    csv = df.to_csv(index=False, encoding='utf-8-sig')
    b64 = base64.b64encode(csv.encode(encoding='utf-8-sig')).decode(encoding='utf-8-sig')
    href = f'<a href="data:file/csv;base64,{b64}" download="trends.csv">Download CSV File</a>'
    return href

st.write('Download a table') 
st.markdown(filedownload(df), unsafe_allow_html=True)
st.write("""***""")




