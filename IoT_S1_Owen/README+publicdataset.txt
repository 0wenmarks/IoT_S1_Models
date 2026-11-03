IoT S1:Human activity recognition


Public Dataset:https://www.kaggle.com/datasets/nickpisarevds/smartphonebased-recognition-of-hapt

Collected dataset: https://bathspaonline-my.sharepoint.com/shared?id=%2Fpersonal%2Ff%5Fnwebonyi%5Fbathspa%5Fac%5Fuk%2FDocuments%2FSecuring%20The%20IoT%202025%2D2026%20S1%20Dataset%2FKeolan%2C%20Ollie%2C%20Owen%2C%20Zico%2C%20Sam&listurl=%2Fpersonal%2Ff%5Fnwebonyi%5Fbathspa%5Fac%5Fuk%2FDocuments&viewid=810fa116%2D6b59%2D4774%2Db418%2D0ac05aa00569&d=wa19fc1b5c0254de8baffb0d463d9147e&csf=1&CT=1793725557514&OR=OWA%2DNT%2DMail&SI=NonSentItems&clickParams=eyJYLUFwcE5hbWUiOiJNaWNyb3NvZnQgT3V0bG9vayBXZWIgQXBwIiwiWC1BcHBWZXJzaW9uIjoiMjAyNjEwMzAwMDcuMDQiLCJPUyI6IldpbmRvd3MgMTAifQ%3D%3D&SLSync=N&cidOR=Client&FolderCTID=0x0120001CD4A501ED49EF4CB9D8CA603813AF12

Files in here:

src/pre_process_filter.py: Contains all the python code for the filtering of the data and turning it into one table of 27 features for both HAPT and our own dataset
notebooks/devTest.ipynb: Random Forest and SVM model using the public dataset (SMV dropped as it performed worse than Random Forest)
notebooks/own_data_model.ipynb: Random Forest Model on our group's data, trained on two and tested on one for all participants.


How to run:

1. In a terminal do 'pip install -r requirements.txt
2. Add the HAPT 'RawData' folder from the link at the top to data/hapt_raw
3. Add the 4 collected dataset folders P1,P2,P3,P4 from the link at the top to data/own_data
4. Open each notebook and choose Restart and then Run All


Data:

The data in the data folder has HAPT's 'RawData' folder in hapt_raw and Our own raw dataset from P1-P4 in own_raw. P1-P3 are the 3 used for the Random Forest Test and Train while P4 was used for the robustness test. All data folders were excluded from the repo with .gitignore


#I acknowledge the use of Claude https://claude.ai to assist with code: debugging several elements of this program in addition to providing code recommendations for how to produce the several graphical outputs. All outputs were critically evaluated and modified.
