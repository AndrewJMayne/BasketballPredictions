import numpy as np 
import pandas as pd 
import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import streamlit as st
from PIL import Image


# Load  model a 
model = joblib.load(open("model-v1.joblib","rb"))

def data_preprocessor(df):

    df.season_year = df.season_year.map(round())
    df.birth_year = df.birth_year.map(round())
    return df

def visualize_confidence_level(prediction_proba):

    data = (prediction_proba[0]*100).round(2)
    salary_percentage = pd.DataFrame(data = data,columns = ['Salray Range'],
                                   index = ['$0 - $2,000,000', '$2,000,000 - $5,000,000', '$5,000,000 - $15,000,000', '$15,000,000+'])
    ax = salary_percentage.plot(kind='barh', figsize=(7, 4), color='#722f37', zorder=10, width=0.5)
    ax.legend().set_visible(False)
    ax.set_xlim(xmin=0, xmax=100)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
    
    vals = ax.get_xticks()
    for tick in vals:
        ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

    ax.set_xlabel(" Percentage(%) Confidence Level", labelpad=2, weight='bold', size=12)
    ax.set_ylabel("Salary range", labelpad=10, weight='bold', size=12)
    ax.set_title('Prediction Confidence Level ', fontdict=None, loc='center', pad=None, weight='bold')

    st.pyplot()
    return

st.write("""
# Basketball salary perdiction app
Perdicts the salary range of an nba player using values provied in the sidebar
""")

def get_user_input():
    
    ppg = st.sidebar.slider('points per game', 0, 40.0, 11.3)
    apg = st.sidebar.slider('assists per game', 0.0, 13.0, 3.0 )
    rpg = st.sidebar.slider('rebounds per game',0.0, 18.0, 4.7)
    mpg = st.sidebar.slider('minute per game', 0.0, 48.0, 26.5)
    season_year = st.sidebar.slider('start year of season', 1999, 2022, 2010)
    birth_year = st.sidebar.slider('player birth year',1961, 2004, 1982 )
    
    features = {'PPG': ppg,
            'APG': apg,
            'RPG': rpg,
            'MPG': mpg,
            'season_year': season_year,
            'birth_year': birth_year,
            }
    data = pd.DataFrame(features,index=[0])

    return data

user_input_df = get_user_input()
processed_user_input = data_preprocessor(user_input_df)

st.subheader('User Input parameters')
st.write(user_input_df)

prediction = model.predict(processed_user_input)
prediction_proba = model.predict_proba(processed_user_input)

visualize_confidence_level(prediction_proba)