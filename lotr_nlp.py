# set up and dependencies
import pandas as pd
import re
import os
from sklearn.feature_extraction.text import CountVectorizer #TfidfVectorizer
# from sklearn.pipeline import make_pipeline
# from sklearn.linear_model import LogisticRegression
import plotly
import plotly.express as px
import plotly.graph_objects as go
import json
from flask import Flask, jsonify, render_template
# import nltk
# from nltk.corpus import stopwords
import spacy

app = Flask(__name__)

PATH = os.path.join("..", "data", "files", "headlines_with_nid.csv")

##################################################################

def load_data():
    df = pd.read_csv(PATH)
    df = df[df["lead_paragraph"].notna()]
    return df

##################################################################

def trigram_data():

    data = load_data()

    # Read news article data into dataframe and exclude rows with NaN in "lead_paragraph" column (200+ rows excluded)
    df = pd.read_csv(PATH)
    df = df[df["lead_paragraph"].notna()]

    # Choose which category to analyze for nGrams
    abstracts = df["abstract"]
    headlines = df["abstract"]
    lead = df["lead_paragraph"]

    nltk.download("stopwords")
    stoplist = stopwords.words("english")

    # Get nGrams: (2, 2) for bigrams, (3, 3) for trigrams...
    vectorizer = CountVectorizer(stop_words=stoplist, ngram_range=(3, 3)) # Converts a collection of text documents to a matrix of token counts: the occurrences of tokens in each document. This implementation produces a sparse representation of the counts.
    X = vectorizer.fit_transform(headlines)
    features = vectorizer.get_feature_names()
    print("\n\nX : \n", X.toarray())

    # Getting top ranking features
    sums = X.sum(axis=0)
    data1 = []
    for col, term in enumerate(features):
        data1.append((term, sums[0, col]))
    ranking = pd.DataFrame(data1, columns=["term", "rank"])
    words = ranking.sort_values("rank", ascending=False)
    print("\n\nWords : \n", words.head(20))

    # Select top 50 nGrams and add to new dataframe
    trigram_df = words.head(n=50)

    return trigram_df

##################################################################

# Refactored to work with required JS Plotly format
def trigram_plot():

    df_data = trigram_data()
    term = df_data["term"]
    frequency = df_data["rank"]

    trace1 = {
        "x": term,
        "y": frequency,
        "mode": "markers",
        "hovertemplate":"Trigram: %{x}<br>Count: %{y}<extra></extra>",
        "marker": {
            "color": frequency,
            "size": frequency,
            "sizeref": 0.3,
            "sizemode": 'area',
            "opacity": 1,
        },
    }

    data_to_plot = [trace1,]

    plot_layout = {
        "title": "Trigram frequency",
        "autosize": False,
        "height": 700,
        "width": 1200,
        "margin": {
          "l": 50,
          "r": 50,
          "b": 200,
          "t": 100,
          "pad": 4
        },
        "xaxis": {
            "title": 'Trigrams',
            "automargin": True,
            "tickangle": 45,
            "titlefont": {
                "family": 'Arial, bold',
                "size": 18,
                "color": 'black'
                },
            },
        "yaxis": {
            "title": 'Count',
            "automargin": True,
            # "type": "log",
            "titlefont": {
                "family": 'Arial, sans-serif',
                "size": 18,
                "color": 'black'
                },
            }
    }

    tri_data = json.dumps(data_to_plot, cls=plotly.utils.PlotlyJSONEncoder)
    tri_layout = json.dumps(plot_layout, cls=plotly.utils.PlotlyJSONEncoder)
    fig = go.Figure(data_to_plot, plot_layout)
    plotly.io.write_json(fig, "static/js/trigrams.json")

    return tri_data, tri_layout

##################################################################

def bigram_data():

    data = load_data()

    # Read news article data into dataframe and exclude rows with NaN in "lead_paragraph" column (200+ rows excluded)
    df = pd.read_csv(PATH)
    df = df[df["lead_paragraph"].notna()]

    # Choose which category to analyze for nGrams
    abstracts = df["abstract"]
    headlines = df["abstract"]
    lead = df["lead_paragraph"]

    nltk.download("stopwords")
    stoplist = stopwords.words("english")

    # Get nGrams: (2, 2) for bigrams, (3, 3) for trigrams...
    vectorizer = CountVectorizer(stop_words=stoplist, ngram_range=(2, 2)) # Converts a collection of text documents to a matrix of token counts: the occurrences of tokens in each document. This implementation produces a sparse representation of the counts.
    X = vectorizer.fit_transform(headlines)
    features = vectorizer.get_feature_names()
    print("\n\nX : \n", X.toarray())

    # Getting top ranking features
    sums = X.sum(axis=0)
    data1 = []
    for col, term in enumerate(features):
        data1.append((term, sums[0, col]))
    ranking = pd.DataFrame(data1, columns=["term", "rank"])
    words = ranking.sort_values("rank", ascending=False)
    print("\n\nWords : \n", words.head(20))

    # Select top 50 nGrams and add to new dataframe
    bigram_df = words.head(n=50)

    return bigram_df

##################################################################

# Refactored to work with required JS Plotly format
def bigram_plot():

    df_data = bigram_data()
    term = df_data["term"]
    frequency = df_data["rank"]

    trace1 = {
        "x": term,
        "y": frequency,
        "mode": "markers",
        "hovertemplate":"Bigram: %{x}<br>Count: %{y}<extra></extra>",
        "marker": {
            "color": frequency,
            "size": frequency,
            "sizeref": 0.9,
            "sizemode": 'area',
            "opacity": 1,
        },
    }

    data_to_plot = [trace1,]

    plot_layout = {
        "title": "Bigram Frequency",
        "autosize": False,
        "height": 700,
        "width": 1200,
        "margin": {
          "l": 50,
          "r": 50,
          "b": 200,
          "t": 100,
          "pad": 4
        },
        "xaxis": {
            "title": 'Bigrams',
            "automargin": True,
            "tickangle": 45,
            "titlefont": {
                "family": 'Arial, bold',
                "size": 18,
                "color": 'black'
                },
            },
        "yaxis": {
            "title": 'Count',
            "automargin": True,
            # "type": "log",
            "titlefont": {
                "family": 'Arial, sans-serif',
                "size": 18,
                "color": 'black'
                },
            }
    }

    bi_data = json.dumps(data_to_plot, cls=plotly.utils.PlotlyJSONEncoder)
    bi_layout = json.dumps(plot_layout, cls=plotly.utils.PlotlyJSONEncoder)
    fig = go.Figure(data_to_plot, plot_layout)
    plotly.io.write_json(fig, "static/js/bigrams.json")

    return bi_data, bi_layout

##################################################################

trigram_plot()
bigram_plot()

##################################################################

# For reference, this is Ed's example for flask app, from p6w-6-python-only (app.py).
# Using this as a development app to test deployment and see plot results.

@app.route("/")
def home():
    tri_data, tri_layout = trigram_plot()
    return render_template("index.html", data=tri_data, layout=tri_layout)

@app.route("/bigrams")
def bigram():
    bi_data, bi_layout = bigram_plot()
    return render_template("bigram.html", data=bi_data, layout=bi_layout)

if __name__ == "__main__":
    app.run(debug=True)