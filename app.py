from tkinter import Y
import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
import _tkinter

st.title("Fruits Classifier")
file = st.file_uploader("Rasm yuklash", type=["jpg", "png", "jpeg", "ljpg", 'fjpg'])

if file:
    st.image(file)

    img = PILImage.create(file)

    model = load_learner("./fruits_model.pkl")

    # predict

    pred, pred_id, probs = model.predict(img)


    st.success(f'Bashorat: {pred}')
    st.info(f'Extimolligi: {probs[pred_id]}')

    # plot
    fig = px.pie(values=probs , names = ['Apple', 'Banana', 'Lemon', 'Orange', 'Pomegranate', 'Strawberry'], title = 'Fruits')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(title="Extimolliklar")
    st.write(fig)
