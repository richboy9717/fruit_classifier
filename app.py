import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px

st.header("Fruits Classifier")

st.write("Ushubu model bir nechta mevalarni, xususan Olma, anor, banan, limon, apelsin va Qulupnayni taniy oladi")

st.subheader("⬇️ O'zingiz sinab ko'ring ⬇️")
file = st.file_uploader("Rasm yuklash", type=["jpg", "png", "jpeg", "ljpg", 'fjpg'])

if file:
    st.image(file)

    img = PILImage.create(file)

    model = load_learner("./fruits_model.pkl")

    # predict

    pred, pred_id, probs = model.predict(img)


    st.success(f'Bashorat: {pred} ✅')
    st.info(f'Aniqligi: {probs[pred_id]}')

    # plot
    fig = px.pie(values=probs , names = ['Apple', 'Banana', 'Lemon', 'Orange', 'Pomegranate', 'Strawberry'], title = 'Fruits')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(title="Extimolliklar")
    st.write(fig)
