import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="방송매체 데이터 모델",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="auto",
)

st.title("Hello World!")
st.write("Hello, Streamlit!")

if st.button("Refresh"):
    st.rerun()
