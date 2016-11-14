
from explore.exploredata import ExploreData
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler

class PrepareData(ExploreData):
    def __init__(self):
        ExploreData.__init__(self)
        return
    def __normalize(self):
        self.X_train = self.X_train.astype(np.float32)
        self.X_val = self.X_val.astype(np.float32)
        self.X_test = self.X_test.astype(np.float32)
        
        mean_image = np.mean(self.X_train, axis=0)
        std_image = np.std(self.X_train, axis=0)
        
        self.X_train = (self.X_train - mean_image)/std_image
        self.X_val = (self.X_val - mean_image)/std_image
        self.X_test = (self.X_test - mean_image)/std_image
        
        
#         self.X_test -= mean_image
#         self.X_train = ((self.X_train - 128)/128.0)
#         self.X_val = ((self.X_val - 128)/128.0)
#         self.X_test = ((self.X_val - 128)/128.0)
        
#         sc = MinMaxScaler()
#         sc.fit(self.X_train)
#         self.x_train= sc.transform(self.X_train)
#         self.X_val= sc.transform(self.X_val)
#         self.X_test= sc.transform(self.X_test)
        return
    def __split_dataset(self, train_data):
        # split train data into train and val
        X = train_data[:,:-1]
        y = train_data[:,-1]
        
        
        split = StratifiedShuffleSplit(y, 1, test_size=0.2, random_state=43)
        for train_index, val_index in split:
            self.X_train = X[train_index]
            self.y_train = y[train_index]
            self.X_val =  X[val_index]
            self.y_val =  y[val_index]
        return
    def get_train_validationset_3d(self):
        self.get_train_validationset()
        #hard coded logic
        self.X_train = self.X_train.reshape(-1, 32,32,3)
        self.X_val = self.X_val.reshape(-1, 32,32,3)
        self.X_test = self.X_test.reshape(-1, 32,32,3)
        return self.X_train, self.y_train,self.X_val,self.y_val, self.X_test,self.y_test
        return
    def get_train_validationset(self):
        train_data, test_data = self.get_train_test_data()
        self.__split_dataset(train_data)
        self.X_test = test_data[:,:-1]
        self.y_test = test_data[:,-1]
        self.__normalize()
        self.y_train = self.y_train.reshape(-1, 1)
        self.y_val = self.y_val.reshape(-1, 1)
        self.y_test = self.y_test.reshape(-1, 1)
        return self.X_train, self.y_train,self.X_val,self.y_val, self.X_test,self.y_test
    def run(self):
        self.get_train_validationset()
        return
    

if __name__ == "__main__":   
    obj= PrepareData()
    obj.run()