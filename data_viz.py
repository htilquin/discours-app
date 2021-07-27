import os
from os import path
from PIL import Image
import pandas as pd
import numpy as np
import streamlit as st

import fasttext

from wordcloud import WordCloud, get_single_color_func
from stop_words import get_stop_words

import matplotlib.pyplot as plt


# LOGO OUEST FRANCE
# get data directory (using getcwd() is needed to support running example in generated IPython notebook)
d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
logo_mask = np.array(Image.open(path.join(d, "ouest-france.png")))

model = fasttext.load_model('model_ova.ftz')
df_train = pd.read_pickle("df_train_app.pkl")
df_test = pd.read_pickle("df_test_app.pkl")

class SimpleGroupedColorFunc(object):
    """Create a color function object which assigns EXACT colors
       to certain words based on the color to words mapping

       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.

       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.word_to_color = {word: color
                              for (color, words) in color_to_words.items()
                              for word in words}

        self.default_color = default_color

    def __call__(self, word, **kwargs):
        return self.word_to_color.get(word, self.default_color)


class GroupedColorFunc(object):
    """Create a color function object which assigns DIFFERENT SHADES of
       specified colors to certain words based on the color to words mapping.

       Uses wordcloud.get_single_color_func

       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.

       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.color_func_to_words = [
            (get_single_color_func(color), set(words))
            for (color, words) in color_to_words.items()]

        self.default_color_func = get_single_color_func(default_color)

    def get_color_func(self, word):
        """Returns a single_color_func associated with the word"""
        try:
            color_func = next(
                color_func for (color_func, words) in self.color_func_to_words
                if word in words)
        except StopIteration:
            color_func = self.default_color_func

        return color_func

    def __call__(self, word, **kwargs):
        return self.get_color_func(word)(word, **kwargs)
    
def words_in_cloud(corpus, text, default_color="grey", highlight_color="red", max_words=500, mask=None, font_path=None, predicted_typo="") :
    # Since the text is small collocations are turned off and text is lower-cased
    stopwords = get_stop_words('fr')
    stopwords.append('c\'est')
    stopwords.append('plus')
       
    wc = WordCloud(random_state=42,
                   collocations=False,
                   width=800, height=400,
                   background_color='white', 
                   normalize_plurals = False,
                   max_font_size=120,
                   max_words=max_words,
                   stopwords=stopwords,
                   mask=mask, contour_width=1, contour_color='darkred',
                   font_path=font_path,
                  ).generate(corpus.lower())
   

    # text to list of words
    words_list = text.lower().split()
    color_to_words = { highlight_color : words_list }

    # Create a color function with multiple tones
    grouped_color_func = GroupedColorFunc(color_to_words, default_color)

    # Apply our color function
    wc.recolor(color_func=grouped_color_func)

    # Plot
    # plt.figure(figsize=(16,9))
    # plt.imshow(wc, interpolation="bilinear")
    # plt.text(s=predicted_typo.upper(), x=300, y=216, ha="center", ma="center", fontsize=18, fontweight="ultralight")
    # plt.axis("off")
    # plt.show()

    fig, ax = plt.subplots(figsize=(16,9))
    ax.imshow(wc, interpolation="bilinear")
    ax.text(s=predicted_typo.upper(), x=300, y=216, ha="center", ma="center", fontsize=18, fontweight="ultralight")
    ax.axis("off")

    st.pyplot(fig)
    
def corpus_from_df(df, typology = "Interview", rows=None):
    all_typo_lines = df[df.Typo_str == typology][:rows]
    
    corpus = ''
    
    for article in all_typo_lines.Text:
        corpus = corpus + article
    
    return corpus

def viz_from_pred(df_train, text_a_predire, model, k=-1, threshold=0.3) :
    prediction_text = model.predict(text_a_predire, k=k, threshold=threshold)
    
    for i in range(len(prediction_text[0])):
        label = prediction_text[0][i].split('__label__')[1]
        rate = prediction_text[1][i]
        st.write("{} : {:.4} %".format(label, rate*100))

        corpus = corpus_from_df(df_train, label)
        words_in_cloud(corpus, text_a_predire, mask=logo_mask, font_path="fonts/LiberationSans-Bold.ttf", predicted_typo=label)
        
    if len(prediction_text[0]) == 0 :
        st.write(f"Pas de prédiction avec cette limite de {threshold*100} %.")
        prediction_text = model.predict(text_a_predire, k=k, threshold=0)
        
        for i in range(len(prediction_text[0])) :
            label = prediction_text[0][i].split('__label__')[1]
            rate = prediction_text[1][i]
            st.write("{} : {:.3} %".format(label, rate*100))
        
    return prediction_text