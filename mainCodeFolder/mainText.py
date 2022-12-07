#Plagiarism Detector for Text-Based CS Assignments
#code was written by Luke Evans

#library imports
import os, glob
import time

import PyPDF2
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# from sklearn import tree
# from sklearn import svm
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.preprocessing import LabelBinarizer
# from sklearn.svm import LinearSVC
# from sklearn.tree import DecisionTreeClassifier

#PDF and DOCX to text Extraction


suffix = '.txt'
extensionPDF = '.pdf'
extensionDOCX = '.docx'
pathPrefix = "."
year = '2021'
profPassword = 'teacher1120'
collectedData = []
runTime = time.time()
s = ''
# pdfDF = pd.DataFrame()
# docxDF = pd.DataFrame()

userType = input('Enter user type (t for teacher, s for student: \n')
if(userType == 't'):
    teacherPassword = input('Enter teacher password: \n')
    for i in range(10):
        if(teacherPassword == profPassword):
            print('Welcome teacher!')
            granularity = input('Check for plagiarism by sentence, by paragraph, or none?\n '
                'Enter s for sentence, p for paragraph, or press enter again to decline prompt \n')
            break
        else:
            print('Wrong Password. ')
elif(userType == 's'):
    studentName = input('Enter student id (example: student1013): \n')
    s = studentName
    print('Welcome student! ')

#list index and root directory declaration
listIndex = -1
source = 'src1'

#list declarations for gathering lists of information
totalFileList = []
assignmentList = []
assignmentStudentList = []
studentText = []
simScoreList = []



#declaration of lists and dataframes for classification
plagiarizedDF = pd.DataFrame()
plagiarizedDFList = []
cleanDFList = []
plagiarizedContentList = []
cleanContent = []
yearList = []
assignList = []


allegedStudentNameList = []
referencedStudentList = []
categoryList = []
scoreList = []
contentList = []


#separators for text granularity
sentenceMark = '.'
paragraphMark = '\n'
#average similarity
avgTotalSimilarity = 0.907

docx = "docx"


# pdfFiles = []
# for filename in os.listdir('.'):
#        if filename.endswith('.pdf'):
#             pdfFiles.append(filename)
# for filename in pdfFiles:
#     pdfFileObj = open(filename, 'rb')
#      pdfRead = pdfFileObj.read()
# pdfFiles.sort(key=str.lower)



pdfWriter = PyPDF2.PdfFileWriter()

for root, dirs, files in os.walk(source):
    for file in files:
        docxName = file.split('_')[1]
        if docxName == docx
            docx =
        if(userType == 't'):
             listIndex = listIndex + 1
             studentName = file.split('_')[0]
             assignmentName = file.split('_')[1]
             assignmentStudentList.append(studentName)
             assignmentList.append(assignmentName)
             totalFileList.append(file)
             #print(totalFileList[listIndex])
             path = os.path.join(root, file)
             file = open(path,'r', encoding='utf-8', errors='ignore')
             #reads file
             fileRead = file.read()
             #lowercase all file contents
             input = fileRead.lower()
             #lowercase all file contents
             studentText.append(input)
        else:
            studentName = file.split('_')[0]
            if(studentName == s):
                path = os.path.join(root, file)
                file = open(path,'r', encoding='utf-8', errors='ignore')
                fileRead = file.read()
                #lowercase all file contents
                input = fileRead.lower()
                #lowercase all file contents
                studentText.append(input)







#increment, amount, and counting variable declarations
amount = (len(assignmentName) - 2) * (len(studentText))

totalCount = 0
studentIncrement = 1
referenceIncrement = 0
#cosine similarity total. Commented out due to lack of necessity after first use
#cosineSimTotal = 0
while totalCount < amount:

    if(referenceIncrement < 19):
        student = studentText[studentIncrement].replace("\n","")
        student1 = student[studentIncrement].replace('\t' , ' ')
        studentFilename = totalFileList[studentIncrement]
        studentInput = [student]
    else:
        referenceIncrement = 0
        studentIncrement = studentIncrement + 1
        student = studentText[studentIncrement].replace('\n',' ')
        studentInput = [student]
        studentFilename = totalFileList[studentIncrement]

    #tf.fit_transform(smallcorp.split('\n'))
    #content.split('\n')

    referenceStudent = studentText[referenceIncrement]
    referenceStudent1 = referenceStudent[referenceIncrement].replace('\t' , ' ')
    referenceInput = [referenceStudent]
    referenceFilename = totalFileList[referenceIncrement]
    referenceIncrement = referenceIncrement + 1

    #cosine similarity vectorizer


    vectorizer = TfidfVectorizer(analyzer = 'word', stop_words= 'english')
    vector1 = vectorizer.fit_transform(studentInput)
    vector2 = vectorizer.transform(referenceInput)

    #determines sentence granularity for vectorization from user input

    if(userType == 't' and granularity == 's'):
        for i in studentInput:
            i.split('.')
        vector1 = vectorizer.fit_transform(studentInput)
        vector2 = vectorizer.transform(referenceInput)

    if(userType == 't' and granularity == 'p'):
        for i in studentInput:
            i.split('\n')
        vectorizer = TfidfVectorizer(analyzer = 'word', stop_words= 'english', use_idf = True)
        vector1 = vectorizer.fit_transform(studentInput)
        vector2 = vectorizer.transform(referenceInput)




    #cosine similarity calculation
    if(studentFilename.split('_')[0] == referenceFilename.split('_')[0]):
        continue
    else:
        similarity_score = cosine_similarity(vector1, vector2)
        #cosineSimTotal = cosineSimTotal + similarity_score
        simScoreList.append(similarity_score)
        if(similarity_score > avgTotalSimilarity):
            contentList.append(student)
            #print(contentList)
            categoryList.append('1')
            #print(categoryList)
            allegedStudentNameList.append(studentFilename.split('_')[0])
            referencedStudentList.append(referenceFilename.split('_')[0])
            yearList.append(year)
            assignList.append(studentFilename.split('_')[1])
            scoreList.append(similarity_score * 100)
        else:
            contentList.append(student)
            #print(studentInput)
            categoryList.append('0')
            allegedStudentNameList.append(studentFilename.split('_')[0])
            referencedStudentList.append(referenceFilename.split('_')[0])
            yearList.append(year)
            assignList.append(studentFilename.split('_')[1])
            scoreList.append(str(similarity_score * 100))

        #print statement no longer needed after first use
        #print("Similarity score between " + studentFilename.split('_')[0] +" and " +
              #referenceFilename.split('_')[0] + " = " + str(similarity_score))
    totalCount = totalCount + 1

#print(contentList)
contentDF = pd.DataFrame({'Content': contentList})
contentDF = contentDF.dropna()

#print(contentDF)
categoryDF = pd.DataFrame({'Category': categoryList})
categoryDF = categoryDF.dropna()


allegedStudentDF = pd.DataFrame(allegedStudentNameList, columns = ['Alleged Student'])
referencedStudentNameDF = pd.DataFrame(referencedStudentList, columns = ['Referenced Student'])
assignmentDF = pd.DataFrame(assignList, columns = ['Assignment'])
yearDF = pd.DataFrame(yearList, columns = ['Year'])
#percentageDF = pd.DataFrame(scoreList, columns= ['Percentage'])


#Update data frame creation

information = [yearList, assignList, allegedStudentNameList, referencedStudentList, scoreList]
update = pd.DataFrame(information)
updateDF = update.transpose()
updateColumns = ['Year', 'Assignment', 'Alleged Student', 'Referenced Student', 'Plagiarism %']
updateDF.columns = updateColumns
updateDF.to_csv('update.csv', index=False, encoding='utf-8')




print(len((categoryList)))
print(len(contentList))





# average similarity metric, commented out for lack of necessity
# this metric is used to classify and determine plagiarism
# avgSimilarity = cosineSimTotal/amount
# print(avgSimilarity)

#check_plagiarismGranularity()

#---------------------------------------------------------------------------------------
#Classification is commented out. It will not functionally execite.
# output = [contentDF, categoryDF]
# total_sample_DF = pd.concat(output)
# totalDF = total_sample_DF.transpose()


# count_vectorizer = CountVectorizer()
# print(totalDF)
# training_term_counts = count_vectorizer.fit_transform(total_sample_DF['Content'].values)
#
# # Fit and train tf-idf transformer
# tfidf = TfidfTransformer()
# X = tfidf.fit_transform(training_term_counts)
# y = total_sample_DF['Guilty']
# lb = LabelBinarizer()
# y = np.array([number[0] for number in lb.fit_transform(y)])
#
# # Compute and print each classifiers' scores
# def compute_scores(algorithm, type, beginning_time):
#     accuracy = cross_val_score(algorithm, X, y, scoring='accuracy', cv=10).mean()
#     accuracy = round(100 * accuracy, 1)
#     precision = cross_val_score(algorithm, X, y, scoring='precision', cv=10).mean()
#     precision = round(100 * precision,1)
#     recall = cross_val_score(algorithm, X, y, scoring='recall', cv=10).mean()
#     recall = round(100 * recall, 1)
#     f1 = cross_val_score(algorithm, X, y, scoring='f1_weighted', cv=10).mean()
#     f1 = round(100 * f1, 1)
#     roc_auc = cross_val_score(algorithm, X, y, scoring='roc_auc', cv=10).mean()
#     roc_auc = round(100 * roc_auc, 1)
#     ending_time = time.time()
#     seconds = round(ending_time-beginning_time,2)
#
#     scores = type + "\t| " + str(accuracy) + "%\t\t| " + str(precision) + "%\t\t| " + str(recall) + "%\t\t| " + str(f1) + "%\t\t| " + str(roc_auc) + "%\t\t| " + str(seconds) + "s"
#
#     print(scores)
#
# # Naïve Bayes classifier
# def nb_classifier():
#     begin = time.time()
#     classifier = MultinomialNB()
#     classifier.fit(X, y)
#     compute_scores(classifier, "Naïve Bayes\t ", begin)
#
# # SVM classifier
# def svm_classifier():
#     begin = time.time()
#     classifier = LinearSVC(random_state=0)
#     classifier.fit(X, y)
#     compute_scores(classifier, "SVM\t\t\t ", begin)
#
# # Random Forest classifier
# def random_forest_classifier():
#     begin = time.time()
#     classifier = RandomForestClassifier(random_state=0)
#     classifier.fit(X, y)
#     compute_scores(classifier, "Random Forest", begin)
#
# # Decision Trees classifier
# def decision_tree_classifier():
#     begin = time.time()
#     classifier = DecisionTreeClassifier(random_state=0)
#     classifier.fit(X, y)
#     compute_scores(classifier, "Decision Tree", begin)
#
# # Print header column
# print("Classifier\t\t| Accuracy\t| Precision\t| Recall\t| F1\t\t| AUC\t\t| Runtime")
# # Run each classifier:
# nb_classifier()
# svm_classifier()
# decision_tree_classifier()
# random_forest_classifier()


#classification()

print("--- %s second run time ---" % (time.time() - runTime)) #just for timing
