# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Installations
Installations required for this include the following Python libraries: [pandas](https://pandas.pydata.org/), [nltk](https://www.nltk.org/), [sqlalchemy](https://www.sqlalchemy.org/), and [scikit-learn](https://scikit-learn.org/stable/). This project also uses the sys, re, and pickle Python modules. The app uses [Flask](http://flask.pocoo.org/) and [Plotly](https://plot.ly/).

## Motivation
The primary motivation for this project was to complete a component of the Udacity Data Scientist Nanodegree. In particular, this component of the course was dedicated to understanding pipelining, including ETL pipelines (done in this project with SQLAlchemy) and machine learning/NLP pipelines (done in this project with sklearn).

## File Descriptions
There are four directories in this repo:
- /app - contains files related to the Flask app
- /data - contains raw data files and files related to the data processing/ETL pipeline
- /models - contains files related to model training
- /notebooks - contains Jupyter notebooks (and related files) used in preparation of the pipelines

## Dataset
The dataset consists of 26248 messages, originally in French, that were communicated in an emergency.
Additionally, we have categories for each message, where each message has been classified as relating
or not relating to any of 36 categories. Examples of categories include `request`, `offer`, `aid_related`,
etc. The goal is to use each message and decide, for each of the 36 categories, which ones they are or 
aren't related to.

### Dataset Imbalance
This dataset is imbalanced, where some categories have relatively few positive examples. This affects
model training in that the model will learn much more successfully how to classify negative examples,
but for some categories, struggle to have high recall. This problem was attempted to be handled by
weighting recall more heavily in the cross-validation scoring; in particular, we used the F4 score to
evaluate models, which means that recall was 4 times more important than precision in the score. Using
this criteria, the model training then focused heavily on trying to identify true positives, which is 
reflected in the model scores.

## Summary
***TO DO*** This project is still in progress - a summary will be added once it's complete!

Baseline performance (random forest classifier with default parameters)
- Overall accuracy: 0.25
- Micro/macro/weighted precision: 0.81 / 0.55 / 0.74
- Micro/macro/weighted recall: 0.48 / 0.18 / 0.48
- Micro/macro/weighted F1 score: 0.60 / 0.23 / 0.54

Model performance:
- Overall accuracy: 0.29
- Micro/macro/weighted precision: 0.81 / 0.56 / 0.75
- Micro/macro/weighted recall: 0.58 / 0.26 / 0.58
- Micro/macro/weighted F1 score: 0.68 / 0.32 / 0.62 
- Micro/macro/weighted F4 score: 0.59 / 0.27 / 0.59

## Authors
The author of most of the repo content is me! Data provided by Figure Eight, linked below in the acknowledgements. Starter code for this project and the repo was provided as part of the Udacity Data Scientist Nanodegree - this includes most of the boilerplate code in the /app directory, as Flask/front-end development was not a focus of this project.

## Acknowledgements
Thanks to [Figure Eight](https://www.figure-eight.com/) for providing the data as a Udacity partner, and thanks to Udacity for including this project as a part of their Data Scientist Nanodegree. I got some cool experience learning about pipelines in SQLAlchemy and scikit-learn.
