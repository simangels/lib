import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class Setting_Up_Csv:
    def __init__(self,path):
        self.path=path

    def read_Data(self):
        self.mydata=pd.read_csv(self.path,index_col=False)
        self.setting_data()

    def setting_data(self):
        pretraining_par, pretesting_par, pretraining_lab, pretesting_lab = train_test_split(self.mydata[["X", "Y"]],self.mydata[["label"]],test_size=0.2)
        parameter1 = np.matrix(pretraining_par["X"])
        parameter2 = np.matrix(pretraining_par["Y"])
        self.merged_matrix = np.concatenate((np.matrix(np.ones(pretraining_par.shape[0])).T, parameter1.T, parameter2.T,),axis=1)
        self.training_lab = np.matrix(pretraining_lab["label"])
        self.theta = np.zeros(self.merged_matrix.shape[1])
        self.theta = self.theta.reshape(self.merged_matrix.shape[1], 1)
        parameter3 = np.matrix(pretesting_par["X"])
        parameter4 = np.matrix(pretesting_par["Y"])
        self.merged_mat_test = np.concatenate((np.matrix(np.ones(parameter3.shape[1])).T, parameter3.T, parameter4.T), axis=1)
        self.testing_lab = np.matrix(pretesting_lab["label"])
        print("ss",self.merged_matrix.shape)



