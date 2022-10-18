#mostly used as a call to cppToTxt.py to test and see if my theory works
import os.path
import pathlib

from cppToTxt import cppToTxt
print("~~~~~~~~")
print (pathlib.Path('student9097.cpp'))
if pathlib.Path('student9097.cpp').suffix == ".cpp":
    with open("student9097.cpp") as fp:
        cppOG= fp.read()
        cppToTxt(cppOG)
