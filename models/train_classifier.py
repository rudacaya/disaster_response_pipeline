import sys
# import libraries
from sqlalchemy import create_engine
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from nltk.stem import WordNetLemmatizer 
import nltk
nltk.download(['punkt', 'wordnet'])

def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(con = engine, table_name = 'InsertTableName')
    X = df.message
    Y = df.drop(['message', 'id', 'genre', 'original'], axis = 1)
    return X, Y, Y.columns



def tokenize(text):
    tokens = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])}
    parameters = {
        'clf__estimator__n_estimators': [20]
        #'clf__estimator__min_samples_split': [2, 10],
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv.best_estimator_


def evaluate_model(model, X_test, y_test, category_names):
    y_pred = pd.DataFrame(model.predict(X_test))
    y_pred.columns = category_names
    y_test.columns = category_names

    for i in y_test.columns:
        print('Columna: {}'.format(i))
        print(classification_report(y_test[i], y_pred[i]))
    pass

def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()