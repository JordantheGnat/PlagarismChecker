# PlagarismChecker
This is a plagarism checker for CSC-3400-01: Artificial Intelligence at Belmont University. We will be creating a plagarism checker 
with capabilities for both CPP and C code

The dataset we used is from https://ieee-dataport.org/open-access/programming-homework-dataset-plagiarism-detection

To get the code running, you will have to download the databse from this site, as it was much too large for github to process, and put it in an order where the main program and the src folder are on the same level, otherwise my processing won't work. My setup is:
![image](https://user-images.githubusercontent.com/71861100/205502585-37e20fa0-d056-4622-8f90-1b297d8413a5.png)

Just so you know, there will be two major file groups when you download the file, using winrar. SRC and Stats. For this, we are only using the SRC folder, and the bottom txt file, the one labeled ground-truth-static-anon.txt, which you will have to put in the SRC folder. Sorr the setup is so complicated, dataset was too big to make it any simpler.

Of if you'd like you could just use this dropbox file https://www.dropbox.com/s/ldt6qnh7l17e5a4/src.zip?dl=0, but I'm not 100% sure how it will work, as the way we all did it was by downloading the dataset then only using the src folder.

Some notes about the organization of this GitHub. Most of the stuff we had in this isn't in the final project, and can be found in the folders codeFolder and FolderOMisc. The main, functioning/mostly functioning code will be in mainCodeFolder.

If you are using a mac, when you get main.py downloaded from PlagarismChecker/mainCodeFolder, and you get the src file in the right spot, then open main.py



Note: We did out best to label the main.py file between Baxter's and Jordan's code, while Luke wrote the entirety of mainText.py

Extra stuff that we used:

Some sources I found for plagarism checking datasets/tutorials

https://github.com/stonecoldnicole/Plagiarism_Detection

https://towardsdatascience.com/build-a-plagiarism-checker-using-machine-learning-6538110ce162

https://ieee-dataport.org/open-access/programming-homework-dataset-plagiarism-detection 


Additions from Luke:

The text duplication detection algorithm is a separate program which can be downloaded from the file name "textPlagiarismChecker.zip" located above
The only folder which is sourced from in this project is 'src1', which contains .txt files to execute the program. The PDF and .DOCX files are included in the "textData.zip" folder in the "textPlagiarismChecker" folder.

Some notes about usage of the program:
The teacher password is "teacher1120" with no quotations when entering into the program.

Link to zip file containing the program: https://drive.google.com/file/d/1yv7GVKxhPQxk7Ub7_cJZx5HrRVJlORT-/view?usp=sharing

Computer Science Assignment Text Dataset (located in zip file named "textData"): 
Original paper pdf (pages 5 through 9 discuss how dataset was created): https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=3f79f4ae839317858d578a11c7454d04253a13ea

-------------------------------------------------------------------------------------------




