import os
import time
import numpy as np
import pandas as pd

# removes warnings from output
import warnings
warnings.filterwarnings('ignore')

from joblib.numpy_pickle_utils import xrange
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

start_time = time.time()
listGroup = [[] for x in xrange(67)]

maxLGLen = 0
assignmentList = []
guiltyStudents = []
for root, subfolders, filenames in os.walk("src"): #walks through the files, grabbing all I need
    for file in filenames:
        path = os.path.join(root, file)

        if file == "ground-truth-static-anon.txt":
            file =  open(path,'r')
            input = file.read()
            input = input.replace(',', '\n')
            guiltyList = input.split()
            listCount = -1
            #above is just file reading
            for i in guiltyList:
                if i not in guiltyStudents and ("A" in i) == False and ("B" in i) == False and ("-" in i)==False:
                    if i not in guiltyStudents: # above checks to see if i is student or class or -
                        guiltyStudents.append(i) #adds i to temp list
            for i in guiltyList:
                if ("A" in i) or ("B" in i)== True :
                    tempString = i.replace("/", '\\')
                    assignmentList.append(tempString) #adds assignment to list of assignments

                    listGroup[listCount].insert(0, tempString) #inserts  the class to beginning of list of lists
                    listGroup[listCount].insert(0,1)    #inserts a 1 before the class
                    q =len(listGroup[listCount])
                    if q > maxLGLen:
                        maxLGLen = q #gets longest list of cheaters
                    listCount = listCount + 1 # increments the list, making it move further down a level
                if ("stu" in i)==True:
                    listGroup[listCount].append(i)
# Break between grabbing guilty students and grabbing their files
allStudents = []
for root, subfolders, filenames in os.walk("src"): #walks through the files, grabbing all I need
    for file in filenames:
        path = os.path.join(root, file)
        size = len(file)
        if file.endswith(".c"):
            student = file[:size - 2] #student is just end of file path - extension
            if student not in allStudents:
                allStudents.append(student)
            if file.endswith(".cpp"):
                student = file[:size - 4]
                if student not in allStudents:
                    allStudents.append(student) #makes a list of all students
#All this does is sort the students into the three groups, all clean and guilty
cleanStudents = list(set(allStudents).difference(guiltyStudents))
cleanStudents = [i for i in cleanStudents if i]
cleanStudents.sort()
allStudents = [i for i in allStudents if i] #  ALl this  above and below is sorting
allStudents.sort()
guiltyStudents = [i for i in guiltyStudents if i]
guiltyStudents.sort()
k = 2
listOfColumns = ["Guilty","Class"]
listOfDropColumns =["Guilty","Class"]
while k <maxLGLen:
    listOfColumns.append(str(k-1)) # makes a big list of columns for the dataframe to be named under
    k = k+1
guiltyAllClassDF=pd.DataFrame(listGroup, columns=listOfColumns)
guiltyAllClassDF.drop(index=guiltyAllClassDF.index[-2], axis=0, inplace=True) #makes dataframe for mostly viewing and understanding
guiltyAllClassDF.drop(index=guiltyAllClassDF.index[-1], axis=0, inplace=True)
processingDataFrame = guiltyAllClassDF.drop( listOfDropColumns, axis=1) #makes dataframe for processing
cleanDataFrame = pd.DataFrame()
guiltyDF = pd.DataFrame()
cleanTest = []
guiltyTest2 = []
assignmentNumber = ""

# year must be in format "A2016", "B2016, "A2017", or "B2017"
# term is "Z1", "Z2", "Z3", "Z4", "Z5", or "Z6"
# assignment is "Z1", "Z2", "Z3", "Z4", "Z5", or "Z6"
# os_version is "mac" or "pc"
def guilty_clean_dataframes_creation(year, term, assignment, os_version):
    separator = "\\"
    if os_version == "mac":
        separator = "/"
    assignment_path = "src" + separator + year + separator + term + separator + assignment

    for root, subfolders, files in os.walk("src"): # Walks through files in src directory
        if (assignment_path in root)==True: # Picks a class to find all the class files in.
            assignmentNumber = year + separator + term + separator + assignment #Takes values from parameter
            assignmentIndex = assignmentList.index(assignmentNumber) #accesses the assignment used based on it's assignmentNumber
            guiltyTest = processingDataFrame.loc[assignmentIndex, :].values.tolist() #Takes the guilty processing df and grabs the assingment from it
            guiltyTest =[i for i in guiltyTest if i is not None] #clears any nones
            guiltyTestLen = len(guiltyTest)
            int = 0
            while int <= (guiltyTestLen-1):
                tempSTR= guiltyTest[int]
                if("A" in assignment_path)==True:
                    tempSTR = (assignment_path + "/" + tempSTR + ".c")#All a assignments end in c
                if("B" in assignment_path)==True:
                    tempSTR = (assignment_path + "/" + tempSTR + ".cpp")#ALl b assignments in CPP
                fileTemp = open(tempSTR,'r', encoding = "utf-8")
                openedFileTemp = fileTemp.read()
                guiltyTest2.append(openedFileTemp)
                int = int+1
            guiltyDF = pd.DataFrame({'student':guiltyTest2})
            for i in files:
                if i not in guiltyTest:
                    path = os.path.join(root, i)
                    #cleanTest.append(i) #appends students intead of paths
                    tempFile = open(path,'r', encoding = "utf-8")
                    tempRead = tempFile.read()
                    cleanTest.append(tempRead) #appends paths for opening to dataframe
            cleanDataFrame = pd.DataFrame({'student':cleanTest})

    cleanDataFrame['Guilty'] = 0
    cleanDataFrame['Assignment'] = assignmentNumber
    guiltyDF['Guilty'] = 1
    guiltyDF['Assignment'] = assignmentNumber

    # Merge guilty and innocent DataFrames
    finalDF = [cleanDataFrame, guiltyDF]
    training_data = pd.concat(finalDF)
    return training_data

total_sample_DF = guilty_clean_dataframes_creation("A2016", "Z2", "Z3", "Mac")
print(total_sample_DF)

# Extract term counts
count_vectorizer = CountVectorizer()
training_term_counts = count_vectorizer.fit_transform(total_sample_DF['student'].values)

# Fit and train tf-idf transformer
tfidf = TfidfTransformer()
X = tfidf.fit_transform(training_term_counts)
y = total_sample_DF['Guilty']
lb = LabelBinarizer()
y = np.array([number[0] for number in lb.fit_transform(y)])

# Compute and print each classifiers' scores
def compute_scores(algorithm, type, beginning_time):
    accuracy = cross_val_score(algorithm, X, y, scoring='accuracy', cv=10).mean()
    accuracy = round(100 * accuracy, 1)
    precision = cross_val_score(algorithm, X, y, scoring='precision', cv=10).mean()
    precision = round(100 * precision,1)
    recall = cross_val_score(algorithm, X, y, scoring='recall', cv=10).mean()
    recall = round(100 * recall, 1)
    f1 = cross_val_score(algorithm, X, y, scoring='f1_weighted', cv=10).mean()
    f1 = round(100 * f1, 1)
    roc_auc = cross_val_score(algorithm, X, y, scoring='roc_auc', cv=10).mean()
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
decision_tree_classifier()
random_forest_classifier()

print("This is just a breakpoint holder")
#Here on out is the part with the classifiers, beware all ye who enter here
print("--- %s second run time ---" % (time.time() - start_time)) #just for timing
