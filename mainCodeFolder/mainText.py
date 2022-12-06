#Plagiarism Detector for Text-Based CS Assignments
#code was written by Luke Evans

#library imports
import os, glob
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_score

from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import pandas as pd


#PDF and DOCX to text Extraction
# print(os.path.join(path, "User/Desktop", "file.txt"))

suffix = '.txt'
extensionPDF = '.pdf'
extensionDOCX = '.docx'
pathPrefix = "."
year = '2021'
profPassword = 'teach1120'
collectedData = []
runTime = time.time()

# pdfDF = pd.DataFrame()
# docxDF = pd.DataFrame()

# userType = input('Enter user type (t for teacher, s for student: \n')
# if(userType == 't')
#     teacherPassword = input('Enter teacher password: \n')
#         while(teacherPassword != profPassword):
#             if(teacherPassword == profPassword):
#                 # print('Welcome teacher!')
#                 #  continue
#             else:
#                 print('Wrong Password. ')
# elif(userType == 's')
#     studentName = input('Enter student id (example: student1013): \n')

# print('Welcome teacher!')
# granularity = input('Check for plagiarism by sentence, by paragraph, or none? \n
# Enter s for sentence, p for paragraph, none')



listIndex = -1
source = 'src1'

#list declarations for gathering lists of necessary information
totalFileList = []
assignmentList = []
assignmentStudentList = []
studentText = []
simScoreList = []
#lists for granularities
sentenceGranularity = []
paragraphGranularity = []


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


for root, dirs, files in os.walk(source):
    for file in files:
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


def check_plagiarism_allNoFilter():
    #increment, amount, and counting variable declarations
    amount = (len(assignmentName) - 2) * (len(assignmentName) - 2)

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

        referenceStudent = studentText[referenceIncrement].replace('\n',' ')
        referenceStudent1 = referenceStudent[referenceIncrement].replace('\t' , ' ')
        referenceInput = [referenceStudent]
        referenceFilename = totalFileList[referenceIncrement]
        referenceIncrement = referenceIncrement + 1

        #cosine similarity vectorizer
        vectorizer = TfidfVectorizer(analyzer = 'word')
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
                scoreList.append(similarity_score * 100)

            #print statement no longer needed after first use
            #print("Similarity score between " + studentFilename.split('_')[0] +" and " +
                  #referenceFilename.split('_')[0] + " = " + str(similarity_score))
        totalCount = totalCount + 1

#print(contentList)
contentDF = pd.DataFrame(contentList, columns = ['Content'])

#print(contentDF)
categoryDF = pd.DataFrame(categoryList, columns = ['Category'])


print(categoryDF)



allegedStudentDF = pd.DataFrame(allegedStudentNameList, columns = ['Alleged Student'])
referencedStudentNameDF = pd.DataFrame(referencedStudentList, columns = ['Referenced Student'])
assignmentDF = pd.DataFrame(assignList, columns = ['Assignment'])
yearDF = pd.DataFrame(yearList, columns = ['Year'])
percentageDF = pd.DataFrame(scoreList, columns= ['Percentage'])


print(len((categoryList)))
print(len(contentList))

#df.values.tolist()
check_plagiarism_allNoFilter()

#Update data frame creation

# information = pd.concat(yearList, assignList, allegedStudentList, referencedStudentList, scoreList]
# update = pd.DataFrame(information)
# updateDF = update.transpose()
#
# updateColumns = ['Year', 'Assignment', 'Alleged Student', 'Referenced Student', 'Plagiarism %']
# update.columns = updateColumns
# update.to_csv('update.csv', index=False, encoding='utf-8')

print(len((categoryList)))
print(len(contentList))

output = [contentList, categoryList]



# average similarity metric, commented out for lack of necessity
# this metric is used to classify and determine plagiarism
# avgSimilarity = cosineSimTotal/amount
# print(avgSimilarity)

#check_plagiarismGranularity()


print(len(contentDF))
print(len(categoryDF))


totalDF = [contentDF, categoryDF]
totalData = pd.concat(totalDF)
totalData.columns = ['Content', 'Category']
def classification():
    accuracyList = []
    precisionList = []
    recallList = []
    f1List = []
    print('Supervised Classification Algorithms for Text Assignment(s)')
    #assignment of plagiarism category of contents in appropriate dataframe
    #declaration of transformed tfidf vectorizer x and category variable y
    v = ''
    total_Data = v.join(contentList)
    data = [total_Data]


    tfidftrans = TfidfVectorizer(analyzer = 'word')
    #total_Data = totalData['Content']
    x = tfidftrans.fit_transform(data)
    y = str(v.join(categoryList))
    print(x)
    #Multinomial Naive Bayes classifier with 10-fold cross validation computation
    nbClf = MultinomialNB()

    #computation of accuracy, precision, recall, and f1 scores
    accuracy = cross_val_score(nbClf, x, y, scoring='accuracy', cv=10)
    accuracyList.append(accuracy)
    print("Naive Bayes Accuracy score: " + str(round(100 * accuracy.mean(), 2)) + "%")
    precision = cross_val_score(nbClf, x, y, scoring='precision', cv=10)
    precisionList.append(precision)
    print("Naive Bayes Precision score: " + str(round(100 * precision.mean(), 2)) + "%")
    recall = cross_val_score(nbClf, x, y, scoring='recall', cv=10)
    recallList.append(recall)
    print("Naive Bayes Recall score: " + str(round(100 * recall.mean(), 2)) + "%")
    f1 = cross_val_score(nbClf, x, y, scoring='f1_weighted', cv=10)
    f1List.append(f1)

    #SVM Classifier with 10-fold cross validation computation
    svmClassifier = svm.SVC()

    #computation of accuracy, precision, recall, and f1 scores
    #print("It takes some time to load SVM classifier data....")
    svmAccuracy = cross_val_score(svmClassifier, x, y, scoring='accuracy', cv=10)
    accuracyList.append(svmAccuracy)
    print("SVM Accuracy score: " + str(round(100 * svmAccuracy.mean(), 2)) + "%")
    svmPrecision = cross_val_score(svmClassifier, x, y, scoring='precision', cv=10)
    precisionList.append(svmPrecision)
    print("SVM Precision score: " + str(round(100 * svmPrecision.mean(), 2)) + "%")
    svmRecall = cross_val_score(svmClassifier, x, y, scoring='recall', cv=10)
    print("SVM Recall score: " + str(round(100 * svmRecall.mean(), 2)) + "%")
    recallList.append(svmRecall)
    svmF1 = cross_val_score(svmClassifier, x, y, scoring='f1_weighted', cv=10)
    print("SVM F1 score: " + str(round(100 * svmF1.mean(), 2)) + "%")
    f1List.append(svmF1)

    #Decision Tree Classifier with 10-fold cross validation computation
    dtClf = tree.DecisionTreeClassifier()

    #computation of accuracy, precision, recall, f1, and auc scores
    #print("It takes some time to load Decision Tree classifier data....")
    dtAccuracy = cross_val_score(dtClf, x, y, scoring='accuracy', cv=10)
    accuracyList.append(dtAccuracy)
    print("Decision Tree Accuracy score: " + str(round(100 * dtAccuracy.mean(), 2)) + "%")
    dtPrecision = cross_val_score(dtClf, x, y, scoring='precision', cv=10)
    precisionList.append(dtPrecision)
    print("Decision Tree Precision score: " + str(round(100 * dtPrecision.mean(), 2)) + "%")
    dtRecall = cross_val_score(dtClf, x, y, scoring='recall', cv=10)
    print("Decision Tree Recall score: " + str(round(100 * dtRecall.mean(), 2)) + "%")
    recallList.append(dtRecall)
    dtF1 = cross_val_score(dtClf, x, y, scoring='f1_weighted', cv=10)
    f1List.append(dtF1)
    print("Decision Tree F1 score: " + str(round(100 * dtF1.mean(), 2)) + "%")
    #Random Forest Classifier with 10-fold cross validation computation
    rfClassifier = RandomForestClassifier()

    #computation of accuracy, precision, recall, and f1 scores.
    #Scores are appended to list

    rfAccuracy = cross_val_score(rfClassifier, x, y, scoring='accuracy', cv=10)
    accuracyList.append(rfAccuracy)
    print("Random Forest Accuracy score: " + str(round(100 * rfAccuracy.mean(), 2)) + "%")
    rfPrecision = cross_val_score(rfClassifier, x, y, scoring='precision', cv=10)
    print("Random Forest Precision score: " + str(round(100 * rfPrecision.mean(), 2)) + "%")
    precisionList.append(rfPrecision)
    rfRecall = cross_val_score(rfClassifier, x, y, scoring='recall', cv=10)
    print("Random Forest Recall score: " + str(round(100 * rfRecall.mean(), 2)) + "%")
    recallList.append(rfRecall)
    rf_F1 = cross_val_score(rfClassifier, x, y, scoring='f1_weighted', cv=10)
    f1List.append(rf_F1)
    print("Random Forest F1 score: " + str(round(100 * rf_F1.mean(), 2)) + "%")




classification()

print("--- %s second run time ---" % (time.time() - runTime)) #just for timing
