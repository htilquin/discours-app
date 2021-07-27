import streamlit as st

from data_viz import *
import pandas as pd

st.title("Typologie de discours")

### SIDE BAR 
st.sidebar.markdown('### Options')
display_viz = st.sidebar.checkbox("Afficher la visualisation", value=True)
seuil = st.sidebar.slider("Seuil de prédiction", min_value=0, max_value=100, value=30, step=10) / 100

#### TEXT AREA : Article à ajouter

help_text_area = "Article en entier, avec son titre."
text = st.text_area("Texte à classer :", help=help_text_area, height=200)
text_clean = text.replace(u'\n', u' ')
nb_mots = len(text.split())
if nb_mots>1 :
    st.write("Texte de {} mots.".format(nb_mots))


### PREDICTIONS

if nb_mots > 1 :
    st.markdown("### Prédictions")
    prediction = viz_from_pred(df_train, text_clean, model, threshold=seuil, display=display_viz)

else :
    st.image(logo_mask, width=50)

st.write("Algorithme utilisé : FastText - Entraînement sur corpus (déséquilibré) d'environ 12 000 articles.")