# Disaster Response Pipeline Project

This project is a simple web application that allows classifying natural disaster messages in real time. The project is divided into three parts:

1. ETL pipeline to clean the data and save it into a SQLite database.
2. Machine learning pipeline used to train a NLP model to classify the message.
3. Web app developed with bootstrap, flask, html and CSS.


## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/database.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/database.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Main Files
- The datasets: disaster_categories.csv and disaster_messages.csv
- ETL Pipeline: data/process_data.py
- ML Pipeline: models/train_classifier.py
- Web App: app/run.py
- html and Bootstrap templates: app/templates/master.html and app/templates/go.html


## Necessary Packages:
- Python 3.8
- Data Wrangling and cleaning libraries: pandas, numpy 
- ML library: scikit-learn
- NLP preprocessing: nltk
- database management: SQLalchemy
- Web app: flask, bootstrap, javascript, plotly
