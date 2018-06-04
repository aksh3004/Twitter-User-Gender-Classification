"""
Course :        Big Data Analytics (CSCI 720)
Project:        Twitter User Gender Classification
Program:        twitter_user_gender_classification.py
Team members:   Akshay Karki (avk1063)
                Sahana Ajit Murthy (sam2738)
Dataset:        gender-classifier-DFE-791531.csv
                (https://www.kaggle.com/crowdflower/twitter-user-gender-classification/data)
ML Toolkit:     Scikit-learn
Python Version: 3.5

This program is used to classify twitter profiles as belonging to a male, a female or a brand with the help
of other profile attributes such as a tweet, profile description, sidebar color, link color, and profile name.

This program has been used for the entire process of Data Cleaning, Feature Extraction, Training and Testing
Classifier models and saving them.

Seven different classifiers have been trained and tested on this data to check which of them gives a better accuracy
on the test dataset. Also, different sets of features have been tested to find out what works best to determine
the gender of the twitter user.
"""

from nltk.corpus import stopwords
import pickle
import string
import re
import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import SGDClassifier
import scipy.sparse as sp
import matplotlib.pyplot as plt


def init():
    """
    This method is used to read and learn about our dataset.
    Also, empty rows are dropped and the rows with target class value missing are dropped too.
    :return: The data set to be used for Predictive Analysis
    """

    # Reading the data from the CSV file using the latin1 encoding.
    data_read = pd.read_csv("gender-classifier-DFE-791531.csv", encoding='latin1')  # Dataset Size = 20050

    # If all the attribute values are empty for any of the rows, we drop them.
    data = data_read.dropna(how='all')          # After dropping, data set size is still 20050

    # Checking the names of the columns/attributes which contains at least one null value
    columns_containing_missing_values = data.columns[data.isnull().any()].tolist()
    print("Column names which has missing values")
    print(columns_containing_missing_values)

    # Since 'gender' is our target variable, we would like to have values for it.
    # So, dropping all the rows which have no values for the 'gender' attribute.
    data = data[data['gender'].notnull()]       # After dropping, dataset size = 19953 rows
    # Also, dropping all the rows which have values as 'unknown' for the 'gender' attribute
    data = data[data['gender'] != 'unknown']    # After dropping, dataset size = 18836 rows

    male_profile_count = len(data[data['gender'] == 'male'])
    print("Male Profile Count " + str(male_profile_count))
    female_profile_count = len(data[data['gender'] == 'female'])
    print("Female Profile Count " + str(female_profile_count))
    brand_profile_count = len(data[data['gender'] == 'brand'])
    print("Brand Profile Count " + str(brand_profile_count))

    return data


def clean_text(text):
    """
    This method is used to clean the text involved in our data set which is mainly the tweet and profile description
    which are known by the attribute names 'text' and 'description'.
    Cleaning involves converting all text to lower case, stripping all punctuation strings before or after every word,
    removing a HTML code, whitespaces, special characters and numbers found in between the text.
    :param text: A string
    :return: Cleaned string
    """
    text = str(text).lower()
    text = text.strip(string.punctuation)
    text = re.sub("&amp;", '', text)
    text = re.sub("https", '', text)
    text = re.sub('\W\s', '', text)
    text = re.sub('\s,\W', '', text)
    text = re.sub('[.!@#$%^&*()_,:;/-]', '', text)
    text = re.sub("\d+", '', text)

    return text


def check_confidence(data):
    """
    This method is used to filter the data based on the gender confidence.
    Gender confidence provides the percentage of confidence the judges had while manually assigning the
    genders to the twitter profiles. So, we consider only those observations in which the gender confidence
    is 100% or 1.
    :param data: The data set to be filtered
    :return: Filtered data set.
    """
    gender_confident_data = data[data['gender:confidence'] == 1]  # Dataset size = 13926
    return gender_confident_data


def split_dataset_for_one_attribute(gender_confident_data, attribute='text_cleaned'):
    """
    This method is used to separate the input and output attributes and divide the data sets into training and testing
    data sets. Also, we only consider one of the attributes as input like the tweet or the profile description
    to determine the gender.
    :param gender_confident_data: The data which has 100% confident gender labels.
    :param attribute: The attribute using which classification model has to be trained and tested.
    :return: The training and testing data sets.
    """

    # Converting text strings into a matrix of word token counts.
    cv = CountVectorizer()
    inputString = cv.fit_transform(gender_confident_data[attribute])

    # Encodes class labels from 0 to Num_of_classes-1
    le = LabelEncoder()
    outputString = le.fit_transform(gender_confident_data['gender'])

    # Splitting the data such that 66% of the data is assigned as training data and the rest as the test data set.
    input_train, input_test, output_train, output_test = train_test_split(inputString, outputString, train_size=0.66)

    return input_train, output_train, input_test, output_test


def combine_tweet_and_description(data):
    """
    This method combines the text from both user's tweet and profile description and forms a new feature
    to be used for classification of user's gender. Also, uses this feature and splits the data set into training
    and testing sets.
    :param data: The data set
    :return: Training and Testing data sets.
    """
    data['combined_text'] = data['text_cleaned'].str.cat(data['description_cleaned'], sep='')
    gender_confident_data = check_confidence(data)

    # Converting text strings into a matrix of word token counts.
    cv = CountVectorizer()
    inputString = cv.fit_transform(gender_confident_data['combined_text'])

    # Encodes class labels from 0 to Num_of_classes-1
    le = LabelEncoder()
    outputString = le.fit_transform(gender_confident_data['gender'])

    # Splitting the data such that 66% of the data is assigned as training data and the rest as the test data set.
    input_train, input_test, output_train, output_test = train_test_split(inputString, outputString, train_size=0.66)
    return input_train, output_train, input_test, output_test


def select_important_features(data):
    """
    This method uses 5 most relevant user profile features that help determine the gender of twitter users.
    We consider the tweet, profile description, sidebar color, link color and profile names.
    :param data: Data set
    :return: Training and Testing data sets.
    """

    selected_attributes = ['text_cleaned', 'description_cleaned', 'sidebar_color', 'link_color', 'name']
    filtered_data = pd.DataFrame(data, columns=selected_attributes)
    output_data = data['gender']

    # Converting text strings into a matrix of word token counts
    cv = CountVectorizer()
    inputString = sp.hstack(filtered_data.apply(lambda attribute: cv.fit_transform(attribute)))

    # Encodes class labels from 0 to Num_of_classes-1
    le = LabelEncoder()
    outputString = le.fit_transform(output_data)

    # Splitting the data such that 66% of the data is assigned as training data and the rest as the test data set.
    input_train, input_test, output_train, output_test = train_test_split(inputString, outputString, train_size=0.66)
    return input_train, output_train, input_test, output_test


def train_and_test_model(In_train, Out_train, In_test, Out_test):
    """
    This method is used to train and test different classifier models for the provided training and testing data sets.
    It also prints out the accuracy and confusion matrices for each classifier.
    :param In_train: Inputs for training
    :param Out_train: Target values for Training
    :param In_test: Inputs for Testing
    :param Out_test: Target values for Testing
    :return: The trained Naive Bayes Classifier for saving the model.
    """

    # Naive Bayes Classifier
    print("Naive Bayes")
    NB_classifier = MultinomialNB()
    NB_classifier.fit(In_train, Out_train)
    predictions = NB_classifier.predict(In_test)
    print(NB_classifier.score(In_test, Out_test))
    NB_Confusion_Matrix = confusion_matrix(Out_test, predictions)
    print(NB_Confusion_Matrix)
    plot_confusion_matrix(NB_Confusion_Matrix)
    print()

    # Stochastic Gradient Descent Classifier
    print("Stochastic Gradient Descent")
    SGD_classifier = SGDClassifier()
    SGD_classifier.fit(In_train, Out_train)
    predictions = SGD_classifier.predict(In_test)
    print(SGD_classifier.score(In_test, Out_test))
    SGD_Confusion_Matrix = confusion_matrix(Out_test, predictions)
    print(SGD_Confusion_Matrix)
    plot_confusion_matrix(SGD_Confusion_Matrix)
    print()

    # MultiLayer Perceptron Classifier
    print("MultiLayer Perceptron")
    MLP_classifier = MLPClassifier()
    MLP_classifier.fit(In_train, Out_train)
    predictions = MLP_classifier.predict(In_test)
    print(MLP_classifier.score(In_test, Out_test))
    MLP_Confusion_Matrix = confusion_matrix(Out_test, predictions)
    print(MLP_Confusion_Matrix)
    plot_confusion_matrix(MLP_Confusion_Matrix)
    print()

    # Random Forest Classifier
    print("Random Forest Classifier")
    RF_classifier = RandomForestClassifier()
    RF_classifier.fit(In_train, Out_train)
    predictions = RF_classifier.predict(In_test)
    scores = cross_val_score(RF_classifier, In_test, Out_test)
    print(scores.mean())
    RF_Confusion_Matrix = confusion_matrix(Out_test, predictions)
    print(RF_Confusion_Matrix)
    plot_confusion_matrix(RF_Confusion_Matrix)
    print()

    # Decision Tree Classifier
    print("Decision Tree")
    DT_classifier = tree.DecisionTreeClassifier()
    DT_classifier.fit(In_train, Out_train)
    predictions = RF_classifier.predict(In_test)
    print(DT_classifier.score(In_test, Out_test))
    DT_Confusion_Matrix = confusion_matrix(Out_test, predictions)
    print(DT_Confusion_Matrix)
    plot_confusion_matrix(DT_Confusion_Matrix)
    print()

    # K-Nearest Neighbors Classifier
    print("K-NN")
    KNN_Classifier = KNeighborsClassifier()
    KNN_Classifier.fit(In_train, Out_train)
    predictions = KNN_Classifier.predict(In_test)
    print(KNN_Classifier.score(In_test, Out_test))
    KNN_Confusion_Matrix = confusion_matrix(Out_test, predictions)
    print(KNN_Confusion_Matrix)
    plot_confusion_matrix(KNN_Confusion_Matrix)
    print()

    # Support Vector Machines
    print("Support Vector Machines")
    SVM_Classifier = svm.SVC()
    SVM_Classifier.fit(In_train, Out_train)
    predictions = KNN_Classifier.predict(In_test)
    print(SVM_Classifier.score(In_test, Out_test))
    SVM_Confusion_Matrix = confusion_matrix(Out_test, predictions)
    print(SVM_Confusion_Matrix)
    plot_confusion_matrix(SVM_Confusion_Matrix)
    print()

    return NB_classifier


def gender_word_counts(data):
    """
    This method is used to determine the top 20 most frequently used words by men, women and brands
    and plot graphs for the same.
    :param data: Original cleaned data set
    :return: None
    """

    # We use the stopwords package from NLTK corpus.
    stop_words = set(stopwords.words('english'))
    data['tweet_words'] = data['text_cleaned'].str.split()
    # Ignoring all the stop words
    data['tweet_words'] = data['tweet_words'].apply(lambda tweet: [word for word in tweet if word not in stop_words])

    # Separating Male, Female and Brand profiles.
    male_profiles = data[data['gender'] == 'male']
    female_profiles = data[data['gender'] == 'female']
    brand_profiles = data[data['gender'] == 'brand']

    print("Top 20 most frequent words used by Men")
    all_male_tweets = ' '.join(male_profiles['tweet_words'].astype(str))
    Male_words = pd.Series(all_male_tweets.split(" ")).value_counts()[:20]
    print(Male_words)
    print()

    print("Top 20 most frequent words used by Women")
    all_female_tweets = ' '.join(female_profiles['tweet_words'].astype(str))
    Female_words = pd.Series(all_female_tweets.split(" ")).value_counts()[:20]
    print(Female_words)
    print()

    print("Top 20 most frequent words used by Brands")
    all_brand_tweets = ' '.join(brand_profiles['tweet_words'].astype(str))
    Brand_words = pd.Series(all_brand_tweets.split(" ")).value_counts()[:20]
    print(Brand_words)

    # Plotting horizontal bar graphs showing Top 20 tweet words used Vs. the word frequency.
    mp = Male_words.plot(kind='barh', stacked=True, colormap='plasma', title="Top 20 most frequently words used by Men")
    mp.set_ylabel("Tweet words used by Males")
    mp.set_xlabel("Word Frequency")
    plt.show()

    fp = Female_words.plot(kind='barh', stacked=True, colormap='plasma',
                           title="Top 20 most frequently words used by Women")
    fp.set_ylabel("Tweet words used by Females")
    fp.set_xlabel("Word Frequency")
    plt.show()

    bp = Brand_words.plot(kind='barh', stacked=True, colormap='plasma',
                          title="Top 20 most frequently words used by Brands")
    bp.set_ylabel("Tweet words used by Brands")
    bp.set_xlabel("Word Frequency")
    plt.show()


def save_model(trained_model):
    """
    Saving a trained classifier model using pickle method.
    :param trained_model: A trained classifier
    :return: None
    """
    twitter_user_gender_classifier_pkl_filename = 'twitter_user_gender_classifier_NB.pkl'
    # Open the file to save as pkl file
    trained_model_pickle_file = open(twitter_user_gender_classifier_pkl_filename, 'wb')
    pickle.dump(trained_model, trained_model_pickle_file)
    # Close the pickle instances
    trained_model_pickle_file.close()


def plot_confusion_matrix(conf_matrix):
    """
    This method is used to plot the confusion matrix for a classifier's predictions.
    :param conf_matrix: The computed confusion matrix
    :return: None
    """
    classes = ['male', 'female', 'brand']
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    threshold = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], 'd'), horizontalalignment="center", color="white"
            if conf_matrix[i, j] > threshold else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_ROC_Curve(FPR, TPR):
    limitRange = [0, 1]
    plt.plot(FPR[2], TPR[2], color='red', lw=2)
    plt.plot(limitRange, limitRange, color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic Curve')
    plt.show()


def main():
    """
    The main method.
    :return: None
    """

    data = init()
    data['text_cleaned'] = [clean_text(tweet) for tweet in data['text']]
    data['description_cleaned'] = [clean_text(line) for line in data['description']]
    print(list(data))
    gender_confident_data = check_confidence(data)

    male_profile_count = len(gender_confident_data[gender_confident_data['gender'] == 'male'])
    print("Male Profile Count " + str(male_profile_count))
    female_profile_count = len(gender_confident_data[gender_confident_data['gender'] == 'female'])
    print("Female Profile Count " + str(female_profile_count))
    brand_profile_count = len(gender_confident_data[gender_confident_data['gender'] == 'brand'])
    print("Brand Profile Count " + str(brand_profile_count))

    X, Y, x, y = split_dataset_for_one_attribute(gender_confident_data)
    train_and_test_model(X, Y, x, y)
    X, Y, x, y = split_dataset_for_one_attribute(gender_confident_data, 'description_cleaned')
    train_and_test_model(X, Y, x, y)

    print("Combining two features")
    X, Y, x, y = combine_tweet_and_description(data)
    train_and_test_model(X, Y, x, y)

    X, Y, x, y = select_important_features(gender_confident_data)
    trained_model = train_and_test_model(X, Y, x, y)
    save_model(trained_model)
    gender_word_counts(data)


if __name__ == "__main__":
    main()
