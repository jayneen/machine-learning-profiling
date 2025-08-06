"""
Authors:
Emily Zapata
Zane Swaims
Evgeniia Nemynova

This program is an ensemble of machine learning models that combines multiple modalities
(images, relational, and text) to classify users by sex, age range, and personality scores.
The program expects jpg files containing profile images, csv files containing relational
likes data, txt files containing profile text, and outputs xml files containing the
classifications associated with each modality as well as the average.
The program utilizes machine learning techniques like neural networks, naive bayes, linear
regressions, and knn.

The dataset used for testing is not provided.
"""

from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
import argparse
import os
import csv
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
import cv2
import joblib
from tensorflow.keras.models import load_model
import xml.etree.ElementTree as ET

RELATION_OUTPUT_DIR = "relation"
TEXT_OUTPUT_DIR = "text"
IMAGE_OUTPUT_DIR = "image"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input')
    parser.add_argument('-o', '--output')

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if not os.path.exists(args.output + RELATION_OUTPUT_DIR + "/"):
        os.makedirs(args.output + RELATION_OUTPUT_DIR + "/")
    if not os.path.exists(args.output + TEXT_OUTPUT_DIR + "/"):
        os.makedirs(args.output + TEXT_OUTPUT_DIR + "/")
    if not os.path.exists(args.output + IMAGE_OUTPUT_DIR + "/"):
        os.makedirs(args.output + IMAGE_OUTPUT_DIR + "/")

    print("Starting relational model...")
    relation_main()
    print("Starting text model...")
    text_main()
    print("Starting image model...")
    image_main()

    print("Calculating results...")
    find_average(args.output, RELATION_OUTPUT_DIR, TEXT_OUTPUT_DIR, IMAGE_OUTPUT_DIR)





def find_average(output_path, RELATION_OUTPUT_DIR, TEXT_OUTPUT_DIR, IMAGE_OUTPUT_DIR):
    """
    Combines all modality outputs and selects the most popular classification.
    :param output_path: File to write to.
    :param RELATION_OUTPUT_DIR: Directory relational outputs are stored in.
    :param TEXT_OUTPUT_DIR: Directory text outputs are stored in.
    :param IMAGE_OUTPUT_DIR: Directory image outputs are stored in.
    """
    # Build paths to the 3 subdirectories
    relational_dir = os.path.join(output_path, RELATION_OUTPUT_DIR)
    text_dir = os.path.join(output_path, TEXT_OUTPUT_DIR)
    image_dir = os.path.join(output_path, IMAGE_OUTPUT_DIR)

    # List all XML filenames (assuming all 3 dirs have the same files)
    filenames = [f for f in os.listdir(relational_dir) if f.endswith(".xml")]

    for filename in filenames:
        # Get full paths to each version of the file
        relational_path = os.path.join(relational_dir, filename)
        text_path = os.path.join(text_dir, filename)
        image_path = os.path.join(image_dir, filename)

        # Parse XML files
        roots = []
        for path in [relational_path, text_path, image_path]:
            tree = ET.parse(path)
            root = tree.getroot()
            roots.append(root)

        # Extract shared user info (assumed identical in all files)
        user_id = roots[0].get("id")

        # Extract gender and age_group from all roots
        genders = [r.get("gender") for r in roots]
        age_groups = [r.get("age_group") for r in roots]

        # Find most common gender and age_group
        majority_gender = Counter(genders).most_common(1)[0][0]
        majority_age_group = Counter(age_groups).most_common(1)[0][0]

        # Extract and average each OCEAN score
        def avg(key):
            return round(sum(float(r.get(key)) for r in roots) / 3.0, 2)

        ext = avg("extrovert")
        neu = avg("neurotic")
        agr = avg("agreeable")
        con = avg("conscientious")
        ope = avg("open")

        # Use the custom method to write the output file
        write_file(user_id, majority_gender, majority_age_group, ope, con, ext, agr, neu, output_path)

# Provided write_file function (unchanged)
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

    # output_file = "".join([output_path, f"{user_id}.xml"])
    output_file = os.path.join(output_path, f"{user_id}.xml")

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


# Map numeric ages to ranges
def map_age_to_range(age):
    try:
        age = float(age)
        if age <= 24:
            return 'xx-24'
        elif 25 <= age <= 34:
            return '25-34'
        elif 35 <= age <= 49:
            return '35-49'
        else:
            return '50-xx'
    except:
        return None


########################################################################################################
###################################         RELATIONAL         #########################################
########################################################################################################

# Load relation data into panda dataframe
def load_relation_data(filepath):
    return pd.read_csv(filepath, names=["userid", "like_id"], low_memory=False)


# Factorize user and like IDs into indices to process the data in the sparse matrix
def factorize_relation(df):
    df['user_index'], user_ids = pd.factorize(df['userid'])
    df['like_index'], like_ids = pd.factorize(df['like_id'])
    return df, user_ids, like_ids


# Create sparse user-like matrix from the user and like dataframe
def create_sparse_matrix(df, num_users, num_likes):
    return csr_matrix(([1] * len(df), (df['user_index'], df['like_index'])), shape=(num_users, num_likes))


# Load profile data and map user IDs
def load_profile_data(profile_path, user_id_map):
    profile_df = pd.read_csv(profile_path) #read csv of profile
    profile_df = profile_df[profile_df['userid'].isin(user_id_map)] #checks if profile is in the provided user-id map
    profile_df['user_index'] = profile_df['userid'].map({uid: idx for idx, uid in enumerate(user_id_map)}) #map the user id
    return profile_df


# Write XML output from the provided information
def relation_write_output(user_id, gender, age_group, traits, output_path):
    gender_str = 'female' if gender == 1 else 'male' #sets the gender into a string from given int
    output_file = os.path.join(output_path, RELATION_OUTPUT_DIR, f"{user_id}.xml") #create XML for given user
    xml_content = "\n".join([f'<user id="{user_id}"',
                             f'age_group="{age_group}"',
                             f'gender="{gender_str}"',
                             f'extrovert="{traits["ext"]:.2f}"',
                             f'neurotic="{traits["neu"]:.2f}"',
                             f'agreeable="{traits["agr"]:.2f}"',
                             f'conscientious="{traits["con"]:.2f}"',
                             f'open="{traits["ope"]:.2f}"',
                             '/>'])
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(xml_content)


# Main execution
def relation_main():
    #read the command line argument provided for the input training data and where it will output the XML files
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    args = parser.parse_args()

    #path for training directory
    TRAINING_DIR = 'training/'
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Load and clean training data from relation csv
    train_rel = load_relation_data(os.path.join(TRAINING_DIR, "relation/relation.csv")) #load training relation data
    like_counts = train_rel['like_id'].value_counts() #counts the likes for each element
    most_liked = like_counts.idxmax()
    least_liked = like_counts.idxmin()
    train_rel = train_rel[~train_rel['like_id'].isin([most_liked])] #remove the most liked

    train_rel, train_user_ids, train_like_ids = factorize_relation(train_rel) #factorize the training data
    x_train_matrix = create_sparse_matrix(train_rel, len(train_user_ids), len(train_like_ids)) #create the training matrix
    #load and clean training data from profile csv
    train_profile = load_profile_data(os.path.join(TRAINING_DIR, "profile/profile.csv"), train_user_ids)
    train_profile['gender'] = pd.to_numeric(train_profile['gender'], errors='coerce')
    train_profile['age_group'] = train_profile['age'].apply(map_age_to_range)
    #drop any profiles that don't have a valid gender or age group
    train_profile = train_profile.dropna(subset=['gender', 'age_group'])
    train_profile['gender'] = train_profile['gender'].astype(int)
    #array of trait name that are used in profile
    trait_names = ['ope', 'con', 'ext', 'neu', 'agr']
    #traverse array to factorize traits in indices
    for trait in trait_names:
        train_profile[trait] = pd.to_numeric(train_profile[trait], errors='coerce')
    #drop any profiles where there are invalid inputs for traits
    train_profile = train_profile.dropna(subset=trait_names)
    train_profile['age_class'], age_labels = pd.factorize(train_profile['age_group'])

    if train_profile.empty:
        print("No valid gender and age-labeled training users found. Exiting.")
        return

    # Train classifiers
    x_train = x_train_matrix[train_profile['user_index'].values]
    y_train_gender = train_profile['gender'].values
    y_train_age = train_profile['age_class'].values

    gender_clf = LogisticRegression(solver='liblinear')
    gender_clf.fit(x_train, y_train_gender)

    age_clf = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000)
    age_clf.fit(x_train, y_train_age)

    trait_models = {}
    for trait in trait_names:
        svr = SVR()
        svr.fit(x_train, train_profile[trait].values)
        trait_models[trait] = svr

    # Load and process test data
    test_rel = load_relation_data(os.path.join(args.input, "relation/relation.csv"))
    test_profile = pd.read_csv(os.path.join(args.input, "profile/profile.csv"))
    test_rel = test_rel[~test_rel['like_id'].isin([most_liked, least_liked])]

    test_user_ids = test_profile['userid'].unique().tolist()
    test_user_index_map = {uid: i for i, uid in enumerate(test_user_ids)}
    like_id_map = {lid: idx for idx, lid in enumerate(train_like_ids)}

    test_rel['user_index'] = test_rel['userid'].map(test_user_index_map)
    test_rel['like_index'] = test_rel['like_id'].map(like_id_map)
    test_rel = test_rel.dropna(subset=['user_index', 'like_index'])
    test_rel['user_index'] = test_rel['user_index'].astype(int)
    test_rel['like_index'] = test_rel['like_index'].astype(int)

    test_matrix = csr_matrix(
        ([1] * len(test_rel), (test_rel['user_index'], test_rel['like_index'])),
        shape=(len(test_user_ids), len(train_like_ids))
    )

    predicted_user_ids = set()
    trait_preds = {}

    if test_matrix.shape[0] > 0:
        # Predict gender and age only for valid users
        gender_preds = gender_clf.predict(test_matrix)
        age_preds = age_clf.predict(test_matrix)
        age_group_preds = age_labels[age_preds]

        # Predict traits for all users in test matrix
        for trait in trait_names:
            trait_preds[trait] = trait_models[trait].predict(test_matrix)

        for idx, user_id in enumerate(test_user_ids):
            if test_matrix[idx].nnz > 0:
                trait_scores = {trait: trait_preds[trait][idx] for trait in trait_names}
                relation_write_output(user_id, gender_preds[idx], age_group_preds[idx], trait_scores, args.output)
                predicted_user_ids.add(user_id)

    # predict attributes for any profile that did not have any overlap with training data
    all_test_user_ids = set(test_profile['userid'])
    fallback_user_ids = all_test_user_ids - predicted_user_ids
    fallback_user_index = {uid: idx for idx, uid in enumerate(test_user_ids)}

    for user_id in fallback_user_ids:
        idx = fallback_user_index[user_id]
        if test_matrix.shape[0] > idx and test_matrix[idx].nnz > 0:
            # Use predicted traits with fallback gender and age
            trait_scores = {trait: trait_preds[trait][idx] for trait in trait_names}
        else:
            # Fully fallback if no usable data
            trait_scores = {
                'ope': 3.50,
                'con': 3.00,
                'ext': 3.25,
                'neu': 2.75,
                'agr': 3.80,
            }
        relation_write_output(user_id, 1, "xx-24", trait_scores, args.output)




########################################################################################################
######################################         TEXT         ############################################
########################################################################################################


""" Path to the directory containing training text files for each user. """
TEXT_DIR = "/home/itadmin/training/text/"
""" Path to the csv file containing training profile information for all users. """
PROFILE_PATH = "/home/itadmin/training/profile/profile.csv"
""" Path to the csv file containing training LIWC features for all users. """
LIWC_PATH = "/home/itadmin/training/LIWC/LIWC.csv"

def text_main():
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
            text_write_file(row['userid'], gender, age, ope, con, ext, agr, neu, args.output)



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

def text_write_file(user_id, gender, age, ope, con, ext, agr, neu, output_path):
    """
        Writes prediction results to an XML file for a given user.
        Args:
            user_id (str): User identifier.
            gender (str): Predicted gender.
            age (str): Predicted age range.
            ope, con, ext, agr, neu (float): Predicted OCEAN scores.
            output_path (str): Directory to save the output file.
    """

    # output_file = "".join([output_path, f"{user_id}.xml"])
    output_file = os.path.join(output_path, TEXT_OUTPUT_DIR, f"{user_id}.xml")

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


########################################################################################################
#####################################         IMAGES         ###########################################
########################################################################################################


# Model image sizes
GENDER_IMAGE_SIZE = 200
AGE_IMAGE_SIZE = 200
OCEAN_IMAGE_SIZE = 200

# Trait order used in the model
OCEAN_TRAITS = ['ope', 'con', 'ext', 'agr', 'neu']

def load_image(image_path, target_size):
    """Loads and preprocesses an image."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.resize(image, (target_size, target_size))
    image = image / 255.0
    return image

def image_write_file(user_id, gender, age, traits, output_path):
    """Creates an XML file with predictions."""
    output_file = os.path.join(output_path, IMAGE_OUTPUT_DIR, f"{user_id}.xml")
    content = "\n".join([
        "<user",
        f"id=\"{user_id}\"",
        f"age_group=\"{age}\"",
        f"gender=\"{gender}\"",
        f"extrovert=\"{traits['ext']:.2f}\"",
        f"neurotic=\"{traits['neu']:.2f}\"",
        f"agreeable=\"{traits['agr']:.2f}\"",
        f"conscientious=\"{traits['con']:.2f}\"",
        f"open=\"{traits['ope']:.2f}\"",
        "/>"
    ])
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

def predict_ocean(image, models):
    """Returns averaged OCEAN predictions from all models."""
    preds = [model.predict(np.expand_dims(image, axis=0), verbose=0)[0] for model in models]
    mean_pred = np.mean(preds, axis=0)
    return dict(zip(OCEAN_TRAITS, mean_pred))

def image_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Input directory with image/ and profile/')
    parser.add_argument('-o', '--output', required=True, help='Output directory for XML files')
    args = parser.parse_args()

    # Create output dir if needed
    os.makedirs(args.output, exist_ok=True)

    # Input paths
    image_dir = os.path.join(args.input, 'image')
    csv_path = os.path.join(args.input, 'profile', 'profile.csv')

    # Load models
    print("Loading models...")
    gender_model = load_model("gender_classifier_cnn_third_try.h5")
    age_model = load_model("age_classifier_balanced_cnn_6532.h5")
    age_encoder = joblib.load("age_label_encoder_frfr.pkl")

    ocean_models = [
        load_model("cnn_model.h5"),
    	load_model("fourth_model.h5"),
    	load_model("resnet50_model.h5"),
    	load_model("efficientnetb0_model.h5"),
    	load_model("inceptionv3_model.h5"),
    	load_model("vgg16_model.h5"),
   	load_model("xception_model.h5"),
    	load_model("densenet121_model.h5")
    ]

    print("Models loaded. Starting prediction...")

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            user_id = row['userid']
            img_path = os.path.join(image_dir, f"{user_id}.jpg")

            # Load images at each model's required size
            img_gender = load_image(img_path, GENDER_IMAGE_SIZE)
            img_age = load_image(img_path, AGE_IMAGE_SIZE)
            img_ocean = load_image(img_path, OCEAN_IMAGE_SIZE)

            if img_gender is None or img_age is None or img_ocean is None:
                print(f"Warning: Image missing or unreadable for user {user_id}")
                continue

            # Gender prediction (binary classification)
            gender_pred = gender_model.predict(np.expand_dims(img_gender, axis=0), verbose=0)[0][0]
            gender = "female" if gender_pred >= 0.5 else "male"

            # Age group prediction (multi-class classification)
            age_probs = age_model.predict(np.expand_dims(img_age, axis=0), verbose=0)[0]
            age_index = np.argmax(age_probs)
            age_group = age_encoder.inverse_transform([age_index])[0]

            # OCEAN ensemble prediction
            traits = predict_ocean(img_ocean, ocean_models)

            # Write output
            image_write_file(user_id, gender, age_group, traits, args.output)

    print("All predictions completed.")



if __name__ == "__main__":
    main()
