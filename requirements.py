import pandas as pd
import streamlit as st
import time
from math import log10


def identity_transform(x):
    return x


def get_predict():
    st.session_state['formVisible'] = False
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()
    st.session_state['resultVisible'] = True


def replace_education(value):
    mapping = {
        'illiterate': 0,
        'basic.4y': 1,
        'basic.6y': 2,
        'basic.9y': 3,
        'high.school': 4,
        'professional.course': 5,
        'university.degree': 6
    }
    return mapping.get(value, value)
