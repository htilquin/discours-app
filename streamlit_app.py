import streamlit as st

from data_viz import *
import pandas as pd

st.title("Typologie de discours")

#### TEXT AREA : Article à ajouter

help_text_area = "Article en entier, avec son titre."
text = st.text_area("Texte à classer :", help=help_text_area, height=200)
nb_mots = len(text.split())
if nb_mots>1 :
    st.write("Texte de {} mots.".format(nb_mots))


### PREDICTIONS

if nb_mots > 1 :
    st.markdown("### Prédictions")
    seuil = st.slider("Seuil de prédiction", min_value=0, max_value=100, value=30, step=10) / 100
    prediction = viz_from_pred(df_train, text, model, threshold=seuil)

else :
    st.image(logo_mask, width=50)