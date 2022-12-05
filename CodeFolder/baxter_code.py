import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

# Read the dataset and use
for root, folders, filenames in os.walk("icpc17"):  # CHANGE TO FILEPATH OF DATASET
    for filename in filenames:
        student = filename.substring(0, filename.indexOf('.'))
        # DataFrame of guilty
        first_instance = True
        if student in guilty_list:                  # Replace guilty_list with Jordan's list of Guilty students
            filepath = os.path.join(root, filename)
            with open(filepath) as txt:
                temporary_guilty = pd.read_csv(filepath, sep="\t", header=None)
                if first_instance:
                    guilty_DF = pd.DataFrame(temporary_guilty)
                    first_instance = False
                else:
                    both_frames = [guilty_DF, temporary_guilty]
                    guilty_DF = pd.concat(both_frames)
        first_instance = True
        # DataFrame of innocent
        if student in innocent_list:                  # Replace innocent_list with Jordan's list of innocent students
            filepath = os.path.join(root, filename)
            with open(filepath) as txt:
                temp_DF = pd.read_csv(filepath, sep="\t", header=None)
                if first_instance:
                    innocent_DF = pd.DataFrame(temp_DF)
                    first_instance = False
                else:
                    frames = [innocent_DF, temp_DF]
                    innocent_DF = pd.concat(frames)

# Relabel to fit classifiers
guilty_DF[1] = "guilty"
innocent_DF[1] = "innocent"
guilty_DF = guilty_DF.rename(columns={0: "File", 1: "Guilty"})
innocent_DF = innocent_DF.rename(columns={0: "File", 1: "Guilty"})
# Merge guilty and innocent DataFrames
final_DF = [guilty_DF, innocent_DF]
training_data = pd.concat(final_DF)

# Extract term counts
count_vectorizer = CountVectorizer()
training_term_counts = count_vectorizer.fit_transform(training_data['File'].values)

# Fit and train tf-idf transformer
tfidf = TfidfTransformer()
X = tfidf.fit_transform(training_term_counts)
y = training_data['Guilty']
lb = LabelBinarizer()
y = np.array([number[0] for number in lb.fit_transform(y)])

# Compute and print each classifiers' scores
def compute_scores(algorithm, type, beginning_time):
    accuracy = cross_val_score(algorithm, X, y, scoring='accuracy', cv=10).mean()
    precision = cross_val_score(algorithm, X, y, scoring='precision', cv=10).mean()
    recall = cross_val_score(algorithm, X, y, scoring='recall', cv=10).mean()
    f1 = cross_val_score(algorithm, X, y, scoring='f1', cv=10).mean()
    roc_auc = cross_val_score(algorithm, X, y, scoring='roc_auc', cv=10).mean()
    accuracy = round(100 * accuracy, 1)
    precision = round(100 * precision,1)
    recall = round(100 * recall, 1)
    f1 = round(100 * f1, 1)
    roc_auc = round(100 * roc_auc, 1)
    ending_time = time.time()
    seconds = round(ending_time-beginning_time,2)

    scores = type + "\t| " + str(accuracy) + "%\t\t| " + str(precision) + "%\t\t| " + str(recall) + "%\t\t| " + str(f1) + "%\t\t| " + str(roc_auc) + "%\t\t| " + str(seconds) + "s"

    print(scores)

# Naïve Bayes classifier
def nb_classifier():
    begin = time.time()
    classifier = MultinomialNB()
    classifier.fit(X, y)
    compute_scores(classifier, "Naïve Bayes\t ", begin)

# SVM classifier
def svm_classifier():
    begin = time.time()
    classifier = LinearSVC(random_state=0)
    classifier.fit(X, y)
    compute_scores(classifier, "SVM\t\t\t ", begin)

# Random Forest classifier
def random_forest_classifier():
    begin = time.time()
    classifier = RandomForestClassifier(random_state=0)
    classifier.fit(X, y)
    compute_scores(classifier, "Random Forest", begin)

# Decision Trees classifier
def decision_tree_classifier():
    begin = time.time()
    classifier = DecisionTreeClassifier(random_state=0)
    classifier.fit(X, y)
    compute_scores(classifier, "Decision Tree", begin)

# Print header column
print("Classifier\t\t| Accuracy\t| Precision\t| Recall\t| F1\t\t| AUC\t\t| Runtime")
# Run each classifier:
nb_classifier()
svm_classifier()
random_forest_classifier()
# decision_tree_classifier()
