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
global maxLGLen
tempList = []
listGroup = [[] for x in xrange(67)]
#print(os.listdir("src"))
def main():
    maxLGLen = 0
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
                listCount = 0
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
                student = file[:size - 2]
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
    allStudents = [i for i in allStudents if i]
    allStudents.sort()
    guiltyStudents = [i for i in guiltyStudents if i]
    guiltyStudents.sort()
    k = 2
    listOfColumns = ["Guilty","Class"]
    listOfDropColumns =["Guilty","Class"]
    while k <maxLGLen:
        listOfColumns.append(str(k-1))
        k = k+1
    print(guiltyStudents)

    listCount2 = 0
    finalGuiltyDataFrame=pd.DataFrame(listGroup, columns=listOfColumns)
    finalGuiltyDataFrame.drop(index=finalGuiltyDataFrame.index[-2], axis=0, inplace=True)
    finalGuiltyDataFrame.drop(index=finalGuiltyDataFrame.index[-1], axis=0, inplace=True)
    processingDataFrame = finalGuiltyDataFrame.drop( listOfDropColumns, axis=1)
    print(finalGuiltyDataFrame)
    cleanDataFrame = pd.DataFrame()
    for root, subfolders, files in os.walk("src"): #walks through the files, grabbing all I need
        for folders in files:
            path = os.path.join(root, file)
            if (root  == 'src\\A2016'):
                for subsubfolders in folders:
                    print()
       #     if (root == 'src\\A2017'):
      #      if (root == 'src\\B2016'):
      #      if (root == 'src\\B2017'):





main()
print("--- %s second run time ---" % (time.time() - start_time)) #just for timing
