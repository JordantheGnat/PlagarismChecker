from typing import List
import numpy as np
import pandas
import pandas as pd
from joblib.numpy_pickle_utils import xrange
import textract
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from random import *
import time
import os
from pathlib import Path
guilty = pd.DataFrame()
classList = []
start_time = time.time()
global guiltyFileDict
global listGroup
tempList = []
listGroup = [[] for x in xrange(67)]
#print(os.listdir("src"))
def main():
    assignmentList=[]
    guiltyStudents = []
    for root, subfolders, filenames in os.walk("src"): #walks through the files, grabbing all I need
        for file in filenames:
            path = os.path.join(root, file)

            if file == "ground-truth-static-anon.txt":
                file =  open(path,'r')
                input = file.read()
                input = input.replace(',', '\n')
                guiltyList = input.split()
                listCount,maxLGLen = 0,0

                for i in guiltyList:
                    if i not in guiltyStudents and ("A" in i) == False and ("B" in i) == False:
                        if i not in guiltyStudents:
                            guiltyStudents.append(i)
                    if ("A2016\Z1\Z1" in i)==True:
                        b = "A2016\Z1\Z1"

                    if ("stu" in i)==True:
                            listGroup[listCount].append(i)

                for i in guiltyList:
                    if ("A" in i) or ("B" in i)== True :
                        tempString = i.replace("/","\\")
                        assignmentList.append(tempString)

                        listGroup[listCount].insert(0, tempString)
                        q =len(listGroup[listCount])
                        if q > maxLGLen:
                            maxLGLen = q
                        listCount = listCount + 1
                    if ("stu" in i)==True:
                        listGroup[listCount].append(i)
                #print(classList)#list of classes
    print(listGroup)
    #print(len(listGroup))
    #print(maxLGLen)
    listCount = 0
    # Break between grabbing guilty students and grabbing their files
    allStudents = []
    for root, subfolders, filenames in os.walk("src"): #walks through the files, grabbing all I need
        for file in filenames:
            path = os.path.join(root, file)
            size = len(file)
            if file.endswith(".c"):
                student = file[:size - 2]
                if student not in allStudents:
                    allStudents.append(student)
                if file.endswith(".cpp"):
                    student = file[:size - 4]
                    if student not in allStudents:
                        allStudents.append(student) #makes a list of all students
    cleanStudents = list(set(allStudents).difference(guiltyStudents))
    cleanStudents = [i for i in cleanStudents if i]
    cleanStudents.sort()
    allStudents = [i for i in allStudents if i]
    allStudents.sort()
    k = 2
    listOfColumns = ["Guilty","Class"]
    listOfDropColumns =["Guilty","Class"]
    while k <maxLGLen:
        listOfColumns.append(str(k-1))
        k = k+1


    listCount2 = 0
    finalGuiltyDataFrame=pd.DataFrame(listGroup, columns=listOfColumns)
    cleanDataFrame = pd.DataFrame()
    finalGuiltyDataFrame.drop(index=finalGuiltyDataFrame.index[-2], axis=0, inplace=True)
    finalGuiltyDataFrame.drop(index=finalGuiltyDataFrame.index[-1], axis=0, inplace=True)
    processingDataFrame = finalGuiltyDataFrame.drop( listOfDropColumns, axis=1)
    print(processingDataFrame)




main()
print("--- %s second run time ---" % (time.time() - start_time)) #just for timing
