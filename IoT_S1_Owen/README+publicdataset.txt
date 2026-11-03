IoT S1:Human activity recognition


Public Dataset:https://www.kaggle.com/datasets/nickpisarevds/smartphonebased-recognition-of-hapt


Files in here:

src/pre_process_filter.py: Contains all the python code for the filtering of the data and turning it into one table of 27 features for both HAPT and our own dataset
notebooks/devTest.ipynb: Random Forest and SVM model using the public dataset (SMV dropped as it performed worse than Random Forest)
notebooks/own_data_model.ipynb: Random Forest Model on our group's data, trained on two and tested on one for all participants.


How to run:

1. In a terminal do 'pip install -r requirements.txt
2. Open each notebook and choose Restart and then Run All


Data:

The data in the data folder has HAPT's 'RawData' folder in hapt_raw and Our own raw dataset from P1-P4 in own_raw. P1-P3 are the 3 used for the Random Forest Test and Train while P4 was used for the robustness test. All data folders were excluded from the repo with .gitignore


#I acknowledge the use of Claude https://claude.ai to assist with code: debugging several elements of this program in addition to providing code recommendations for how to produce the several graphical outputs. All outputs were critically evaluated and modified.
