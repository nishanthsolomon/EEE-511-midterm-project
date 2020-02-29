# Mount-Rainer : Disappointment Cleaver

Steps to Train

1. Install the requirements as in requirements.txt
2. The input data given for the testing needs to be in a csv file with the colums in the following order: Success Rate,Temperature AVG,Relative Humidity AVG,Wind Speed Daily AVG. 
3.python dataprocessing.py to create FinalData_all.csv which contains the final dataset after preprocessing
4. python data_preprocessing.py to create the dataset from the kaggle dataset downloaded in the ./dataset folder.
5. Change the parameters in midterm_project.conf as required.
6. python midterm_project.py to train and evaluate the linear regression model, polynomial regression model and mlp model. This would print the RMSE models for the models. 
