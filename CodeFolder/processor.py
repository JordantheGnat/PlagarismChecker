#Simple translator file for calling in different places
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
guiltyFilesList = pd.DataFrame
#print(os.listdir("src"))
def processor():
    for root, subfolders, filenames in os.walk("src"): #walks through the files, grabbing all I need
        for file in filenames:
            path = os.path.join(root, file)
            if file == "ground-truth-anon.txt":
                file =  open(path,'r')
                input = file.read()
                input = input.replace(',', '\n')
                fileAsList = input.split()
                for i in fileAsList:
                    if "A" in i:
                        classList.append(i)
                        fileAsList.remove(i)
                print(classList)#list of classes
                for i in fileAsList:
                    if "-" in i:
                        fileAsList.remove(i)

                #print(fileAsList) #List of students
    # Break between grabbing guilty students and grabbing their files
    guiltyFilesListCPP = []
    guiltyFilesListC = []
    listOLists = []
    listOLists.append(0)
    print(guiltyFilesList)
    for root, subfolders, filenames in os.walk("src"): #walks through the files, grabbing all I need
        for file in filenames:
            path = os.path.join(root, file)
            size = len(file)
            if file.endswith(".c"):
                potentialGuilty = file[:size - 2]
                if potentialGuilty in fileAsList:
                    for miniList in listOLists:
                        if potentialGuilty in miniList:
                            miniList.append(path)
                        else:
                            listOLists.append(path)
            if file.endswith(".cpp"):
                potentialGuilty = file[:size - 4]
                if potentialGuilty in fileAsList:
                    for miniList in listOLists:
                        if potentialGuilty in miniList:
                            miniList.append(path)
                        else:
                            listOLists.append(path)
                
    i =1000
    while i < 10000:
        
        i=i+1
    print(listOLists)


processor()
