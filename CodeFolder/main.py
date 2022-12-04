#Simple translator file for calling in different places
from typing import List

import numpy as np
import pandas as pd
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
#print(os.listdir("src"))
def main():
    for root, subfolders, filenames in os.walk("src"): #walks through the files, grabbing all I need
        for file in filenames:
            path = os.path.join(root, file)
            if file == "ground-truth-static-anon.txt":
                file =  open(path,'r')
                input = file.read()
                input = input.replace(',', '\n')
                guiltyList = input.split()
                for i in guiltyList:
                    if "A" in i:
                        classList.append(i)
                        guiltyList.remove(i)
                #print(classList)#list of classes
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
    cleanStudents = list(set(allStudents).difference(guiltyList))
    cleanStudents = [i for i in cleanStudents if i]
    cleanStudents.sort()

    guiltyFilesListCPP, guiltyFilesListC, cleanFileListCPP, cleanFileListC = [], [],[],[]
    for root, subfolders, filenames in os.walk("src"): #walks through the files, grabbing all I need
        for file in filenames:
            path = os.path.join(root, file)
            size = len(file)
            if file.endswith(".c"):
                potentialGStudent = file[:size - 2]
                if potentialGStudent in guiltyList:
                    guiltyFilesListC.append(path)
                if potentialGStudent in cleanStudents:
                    cleanFileListC.append(path)

            if file.endswith(".cpp"):
                potentialGStudent = file[:size - 4]
                if potentialGStudent in guiltyList:
                    guiltyFilesListCPP.append(path)
                    3
                if potentialGStudent in cleanStudents:
                    cleanFileListCPP.append(path)
    #print("GUILTY!")
    #print(guiltyFilesListCPP)guiltyFilesListC)
    #print(cleanFileListCPP,cleanFileListC)
    #print("CLEAN!")
    allStudents = [i for i in allStudents if i]
    allStudents.sort()
    listOAssignments = []
    i=1
    j=1
    y=6
    dictIndx = 0
    dict = {}
    tempList = [ ]
    while i <=5:
        for path in cleanFileListC:
            #if ("A201"+str(y)+"\\Z"+str(i)+"\\Z1") in path:

            if ("A201" + str(y) + "\\Z" + str(i) + "\\Z"+str(j)) in path:
                tempList.append(path) #This doesn't work yet, may not get it working in time, once list is cleared it
                #clears it from the dict too 
            else:
                key = ("A201" + str(y) + "\\Z" + str(i) + "\\Z"+str(j))

                dict [str(key)]=tempList
        j=1
        i = i+1

    allCPP = guiltyFilesListCPP + cleanFileListCPP
    allC = guiltyFilesListC + cleanFileListC
    print(dict.keys())

    guiltyList =[*set(guiltyList)]
    guiltyList.sort()
    #print("All: ",allStudents)
    #print("All Clean: ",cleanStudents)
    guiltyList = [x for x in guiltyList if "-" not in x and "Z" not in x]
    #print("All Guilty: ",guiltyList)


main()

print("--- %s second run time ---" % (time.time() - start_time)) #just for timing
