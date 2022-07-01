# Disaster Response Pipeline Project

This project is a part of Udacity's Data Science Nanodegree. For the final project of the Data Engineering module, I analyzed disaster data from Appen (formerly Figure 8) to build a model for an API that classifies disaster messages, using a data set containing real messages that were sent during disaster events. This included building a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency, and a web app where an emergency worker can input a new message and get classification results in several categories.

The project's sections include:

- An ETL pipeline to extract, clean, and process data, then loads it into a SQLite database.
- An ML pipeline that processes the text messages using NLP and trains a multioutput Random Forest classifier using Grid Search. The final model is evaluated on the test data using precision, recall, and f1 score as metrics, and saved as a pickle file.
- And a web dashboard built using python, flask, plotly, and bootstrap, housing visualizations on the training data, and an app that allows you to input any text message and classify it according to the model, outputing the type of disaster response (if any) is adequate.


## Installations

The jupyter notebook with the original data wrangling steps can be installed through anaconda and was written in python 3.10.4.

Additionally, the web app was built on flask and bootstrap, using heroku to deploy. The python packages used can be found in the requirements.txt file.

The csv files with disaster data were provided by Appen.

## Project Motivation
Building this web app allowed me to put into action the set of skills acquired through the Nanodegree program, particularly data engineering and NLP skills that I've been wanting to practice more often. This project is particularly interesting because I was able to use real data and apply it onto a problem with actual stakes, and which could be helpful for several organizations when taken to a more robust level.

## File Descriptions

```
- app
| - templates
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process
|- disaster_messages.csv  # data to process
|- process_data.py # ETL python script
|- DisasterResponse.db   # database to save clean data to
|- ETL Pipeline Preparation.ipynb   # jupyter notebook with ETL steps / data exploration

- models
|- train_classifier.py # NLP-ML training pipeline
|- classifier.pkl  # saved model / not pushed because of size
|- ML Pipeline Preparation.ipynb   # jupyter notebook with NLP-ML steps / data exploration

- README.md
```
## Results & Improvements

This dataset is imbalanced -- some labels like water have few examples. In fact, there is one class () that has no positive instances. This prevented me from trying out models like Gradient Boosting Classifiers, SVMs, or a special `BalancedRandomForestClassifier` to deal with the imbalance, since they require at least one positive class when sampling. A future iteration of the project might involve dropping this feature, or gathering more data to create positive training data, and therefore gaining the ability to try out more models.

To deal with imbalance on the present iteration, I activated the `class_weight` hyperameter of a Random Forest Classifier, setting it to "balanced" in an effort to sample more data from the underrepresented classes.

I also trained the model on recall as a performance measure, as I valued getting all of the important messages, even if the sacrifice was low-precision and therefore a number of false negatives. This is so that the emergency responders are able to find every disaster situation, though a human might need to validate the results further by scanning the message and verifying the appropiate classification. And interesting future approach might by to train a binary classifier on the "related" feature to detect only relevant messages to disaster response using recall, then train a multioutput classifier to classify only the relevant messages, aiming for better precision in the categorization.

Another improvement might be to advise some organizations to connect to based on the categories that the ML algorithm classifies text into.

To preview the app locally, follows these steps:

### Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

In case you want to run this app locally:

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Licensing & Acknowledgements
Feel free to reuse this code under the MIT License. Special thanks to the Appen and Udacity for the data and project framework.

