import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///data/database.db')
df = pd.read_sql_table('data/data/database.db', engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    #graph 2 data
    df1 = df.copy()
    df1['length'] = df1['message'].apply(lambda x: len(x.split()))
    df2 = df1[['length', 'genre']].groupby('genre').mean().reset_index()
    genres = df2['genre']
    length = df2['length']
    x3 = df1.drop(['id', 'original', 'genre', 'original', 'message', 'length'], axis = 1).columns
    y3 = df1.drop(['id', 'original', 'genre', 'original', 'message', 'length'], axis = 1).sum()
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        
        #graph 2
        {
            'data': [
                Bar(
                    x=genres,
                    y=length
                )
            ],

            'layout': {
                'title': 'Average Amount of Words by Genres',
                'yaxis': {
                    'title': "Average Amount of Words"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        #graph 3
       
        {
            'data': [
                Bar(
                    x=x3,
                    y=y3
                )
            ],

            'layout': {
                'title': 'Amount of messages by Class',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Class"
                }
            }
        }
    ]


    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()