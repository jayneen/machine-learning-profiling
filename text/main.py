"""
This script trains Naive Bayes classifiers to predict gender and age as well as
Linear Regression models to predict OCEAN scores based on individual users' text
status updates.

Author: Evgeniia Nemynova
6/2/2025
"""

import argparse
import os
import csv

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB

""" Path to the directory containing training text files for each user. """
TEXT_DIR = "/home/itadmin/training/text/"
""" Path to the csv file containing training profile information for all users. """
PROFILE_PATH = "/home/itadmin/training/profile/profile.csv"
""" Path to the csv file containing training LIWC features for all users. """
LIWC_PATH = "/home/itadmin/training/LIWC/LIWC.csv"

def main():
    """
        Main function to train models and generate predictions based on input data.
        Handles argument parsing, data loading, model training, and output generation.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input')
    parser.add_argument('-o', '--output')

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    text_dir = args.input + "text/"
    liwc_file = args.input + "LIWC/LIWC.csv"
    csv_file = args.input + "profile/profile.csv"

    df = load_data(PROFILE_PATH, TEXT_DIR)

    # split data to train and test
    all_Ids = np.arange(len(df))
    train_Ids = all_Ids[:]

    data_train = df.iloc[train_Ids]

    # process text
    count_vect = CountVectorizer(max_df=0.95, min_df=2)
    train_text = count_vect.fit_transform(data_train['text'])

    # train classifiers
    train_gender = data_train['gender']
    gender_clf = train_classifier(train_text, train_gender)
    train_age = data_train['age']
    age_clf = train_classifier(train_text, train_age)

    # train regressions
    ope_reg = train_regression(data_train['ope'], data_train['liwc'])
    con_reg = train_regression(data_train['con'], data_train['liwc'])
    ext_reg = train_regression(data_train['ext'], data_train['liwc'])
    agr_reg = train_regression(data_train['agr'], data_train['liwc'])
    neu_reg = train_regression(data_train['neu'], data_train['liwc'])

    # for each test user
    with (open(csv_file, newline='', encoding='utf-8') as csvfile):
        reader = csv.DictReader(csvfile)
        for row in reader:
            # read text
            text = read_text(text_dir, row['userid'] + ".txt")
            # read liwc
            liwc = pd.read_csv(liwc_file)[lambda df: df['userId'] == row['userid']].iloc[0].loc['WC':].astype(float).to_numpy()
            f = pd.DataFrame({'text': text, 'liwc': [liwc]})
            # predict statistics
            gender, age, ope, con, ext, agr, neu = predict_stats(gender_clf, age_clf, ope_reg, con_reg, ext_reg, agr_reg, neu_reg, count_vect, f)
            # print predictions
            write_file(row['userid'], gender, age, ope, con, ext, agr, neu, args.output)



def load_data(profile_path, text_dir):
    """
        Loads and merges user profile, LIWC features, and text data into a single DataFrame.
        Args:
            profile_path (str): Path to the profile.csv file.
            text_dir (str): Directory containing user text files.
        Returns:
            pd.DataFrame: Combined DataFrame with user attributes and features.
    """
    # read profile.csv into profiles
    profiles = {}
    with open(profile_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            profiles[row['userid']] = {
                'gender': row['gender'],
                'age': row['age'],
                'ope': row['ope'],
                'con': row['con'],
                'ext': row['ext'],
                'agr': row['agr'],
                'neu': row['neu']
            }

    # read LIWC.csv into user_features
    user_features = {}
    with open(LIWC_PATH, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        for row in reader:
            user_id = row[0]
            features = np.array(row[2:], dtype=np.float32)
            user_features[user_id] = features

    # split data into arrays based on text files
    texts = []
    genders = []
    ages = []
    ope_scores = []
    con_scores = []
    ext_scores = []
    agr_scores = []
    neu_scores = []
    liwc_features = []

    for filename in os.listdir(text_dir):
        user_id = filename[:-4]
        user_data = profiles[user_id]

        texts.append(read_text(text_dir, filename))
        genders.append(user_data['gender'])
        ages.append(map_age_to_range(user_data['age']))
        ope_scores.append(float(user_data['ope']))
        con_scores.append(float(user_data['con']))
        ext_scores.append(float(user_data['ext']))
        agr_scores.append(float(user_data['agr']))
        neu_scores.append(float(user_data['neu']))
        liwc_features.append(user_features[user_id])

    # put all data into a dataframe
    df = pd.DataFrame({'text': texts, 'gender': genders, 'age': ages, 'ope': ope_scores,
                       'con': con_scores, 'ext': ext_scores, 'agr': agr_scores, 'neu': neu_scores,
                       'liwc': liwc_features})

    return df

def write_file(user_id, gender, age, ope, con, ext, agr, neu, output_path):
    """
        Writes prediction results to an XML file for a given user.
        Args:
            user_id (str): User identifier.
            gender (str): Predicted gender.
            age (str): Predicted age range.
            ope, con, ext, agr, neu (float): Predicted OCEAN scores.
            output_path (str): Directory to save the output file.
    """

    output_file = "".join([output_path, f"{user_id}.xml"])

    content = "\n".join([
        "<user",
        f"id=\"{user_id}\"",
        f"age_group=\"{age}\"",
        f"gender=\"{gender}\"",
        f"extrovert=\"{ext}\"",
        f"neurotic=\"{neu}\"",
        f"agreeable=\"{agr}\"",
        f"conscientious=\"{con}\"",
        f"open=\"{ope}\"",
        "/>"
    ])

    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(content)

def train_classifier(text, label):
    """
        Trains a Multinomial Naive Bayes classifier on text data.
        Args:
            text (sparse matrix): Feature matrix from CountVectorizer.
            label: Target labels for classification.
        Returns:
            MultinomialNB: Trained classifier.
    """

    clf = MultinomialNB(alpha=0.12)
    clf.fit(text, label)
    return clf

def train_regression(train_scores, train_features):
    """
        Trains a linear regression model on LIWC features to predict OCEAN scores.
        Args:
            train_scores: Target OCEAN scores.
            train_features (pd.Series): LIWC feature vectors.
        Returns:
            LinearRegression: Trained regression model.
    """

    linreg = LinearRegression()
    X = np.vstack(train_features.values)  # Convert to 2D array
    y = train_scores
    linreg.fit(X, y)
    return linreg

def predict_stats(gender_clf, age_clf, ope_reg, con_reg, ext_reg, agr_reg, neu_reg, count_vect, data_test):
    """
        Generates predictions for a user's gender, age, and OCEAN scores.
        Args:
            gender_clf (MultinomialNB): Trained gender classifier.
            age_clf (MultinomialNB): Trained age classifier.
            ope_reg, con_reg, ext_reg, agr_reg, neu_reg (LinearRegression): Trained regression models.
            count_vect (CountVectorizer): Fitted vectorizer for text.
            data_test (pd.DataFrame): Data containing text and LIWC features.
        Returns:
            tuple: Predicted gender, age, and OCEAN scores.
    """

    test_text = count_vect.transform(data_test['text'])

    y_pred_gender = gender_clf.predict(test_text)
    y_pred_age = age_clf.predict(test_text)[0]
    y_pred_ope = ope_reg.predict(np.array([data_test['liwc'].values[0]]))[0]
    y_pred_con = con_reg.predict(np.array([data_test['liwc'].values[0]]))[0]
    y_pred_ext = ext_reg.predict(np.array([data_test['liwc'].values[0]]))[0]
    y_pred_agr = agr_reg.predict(np.array([data_test['liwc'].values[0]]))[0]
    y_pred_neu = neu_reg.predict(np.array([data_test['liwc'].values[0]]))[0]

    if y_pred_gender == '0.0': y_pred_gender = 'male'
    if y_pred_gender == '1.0': y_pred_gender = 'female'

    return y_pred_gender, y_pred_age, y_pred_ope, y_pred_con, y_pred_ext, y_pred_agr, y_pred_neu

def read_text(text_dir, filename):
    """
        Reads a user's text file and returns its content.
        Args:
            text_dir (str): Directory containing text files.
            filename (str): Name of the text file to read.
        Returns:
            str: Text content of the file.
    """

    with open(os.path.join(text_dir, filename), "r", encoding="latin1") as f:
        text = f.read()
    return text

def map_age_to_range(age):
    """
        Converts a numeric age into a predefined age group.
        Args:
            age: User's age.
        Returns:
            str: Age group category.
    """

    age = float(age)

    if age <= 24:
        return 'xx-24'
    elif 25 <= age <= 34:
        return '25-34'
    elif 35 <= age <= 49:
        return '35-49'
    else:
        return '50-xx'


if __name__ == "__main__":
    main()
