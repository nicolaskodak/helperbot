import streamlit as st
import pandas as pd

d = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

csv = convert_df(d)

st.download_button(
    label="Download data",
    data= csv, # "hello world", # csv, # b"hello world",
    mime= 'txt/csv', # 'application/octet-stream',
    file_name= 'text.csv' # 'text.txt', # 'text.csv',
)