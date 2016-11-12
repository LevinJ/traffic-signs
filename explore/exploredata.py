from utility.dumpload import DumpLoad
import numpy as np
from sklearn.preprocessing import scale



class ExploreData:
    def __init__(self):
    
        return
    
    def load(self):
        dump_load = DumpLoad('../data/test_2.p')
        data = dump_load.load()
#         dump_load2 = DumpLoad('../data/test_2.p')
#         dump_load2.dump(data, 2)
#         train_data = train_data['features']
#         img_mean = np.mean(train_data, axis = 0)
#         img_std = np.std(train_data, axis = 0)
#         train_data_np = (train_data-img_mean)/img_std
#         
#         train_data_sk = scale(train_data.reshape(train_data.shape[0], -1))
#         train_data_sk = train_data_sk.reshape(train_data.shape)
#         print(train_data.shape)
        return
        
    def run(self):
        self.load()
        return
    

    
if __name__ == "__main__":   
    obj= ExploreData()
    obj.run()