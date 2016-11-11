from utility.dumpload import DumpLoad


class ExploreData:
    def __init__(self):
    
        return
    def load(self):
        dump_load = DumpLoad('../data/train.p')
        train_data = dump_load.load()
        print train_data.shape
        return
        
    def run(self):
        self.load()
        return
    

    
if __name__ == "__main__":   
    obj= ExploreData()
    obj.run()