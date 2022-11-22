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
print(os.listdir("src"))
for root, subfolders, filenames in os.walk("src"): #walks through the files, grabbing all I need
    for file in filenames:
        path = os.path.join(root, file)

        if file == "ground-truth-anon.txt":
            my_cols = [str(i) for i in range(12)]  # creates 12 columns, the longest amount in the file
            guilty = pd.read_csv(path,
                                               sep="\t|,",
                                               names=my_cols,
                                               header=None,
                                               engine="python")
            #guilty.fillna("deleteThis")
            i = 0
            while i <10:
                j = i+1
               # guilty2 = guilty[str(i)].astype(str)+ guilty [str(j)].astype(str)
                #guilty2.dropna() #this part was an attempt at concating all the columns into one big one

                i = i+1



print(guilty2)        #C:\Users\jorda\PycharmProjects\PlagarismChecker\student9097.cpp
