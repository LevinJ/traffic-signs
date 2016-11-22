from implement.trafficsignmodel import TrafficSignModel

import sys
import os
from __builtin__ import True
sys.path.insert(0, os.path.abspath('..'))

import tensorflow as tf
import numpy as np
import logging
from bokeh.util.logconfig import level
import sys
from utility.tfbasemodel import TFModel
from preprocess.preparedata import PrepareData
import matplotlib.image as mpimg
from utility.dumpload import DumpLoad
from sklearn.preprocessing import OneHotEncoder
from utility.vis_utils import vis_grid_withlabels
from explore.exploredata import ExploreData


class TrafficSignModel_Predict(TrafficSignModel):
    def __init__(self):
        TrafficSignModel.__init__(self)
        self.do_topk5 = False
  
        return
    def load_images(self):
        imagesFolder = '../wildtrafficsign'
        if  not os.path.exists(imagesFolder):
            imagesFolder = './wildtrafficsign'
        imagesFile = os.listdir(imagesFolder)
        imagesFile.sort()
        webimg = np.zeros((len(imagesFile), 32, 32, 3),dtype=np.uint8)
        for i in range(0,len(webimg)):
            image = (mpimg.imread(imagesFolder+'/'+imagesFile[i]))
            webimg[i] = image[0:32,0:32,(0,1,2)]
        webimgylabel = np.array([14,14,14,14,8])
        enc = OneHotEncoder(n_values=43, sparse=False).fit(webimgylabel.reshape(-1,1))

        webimgylabel  = enc.transform(webimgylabel.reshape(-1,1))
        return (webimg, webimgylabel)
    def normalize_images(self, images):
        dumpload = DumpLoad('../data/meanstdimage.pickle')
        mean_image,std_image = dumpload.load()
        mean_image = mean_image.reshape(32,32,3)
        std_image = std_image.reshape(32,32,3)
        images = (images-mean_image)/ std_image
        return images
    def show_predict_result(self,true_lables, predict_labels):
        true_lables = ExploreData().get_label_names(true_lables)
        predict_labels = ExploreData().get_label_names(predict_labels)
        for i in range(len(true_lables)):
            logging.debug("image {} :True {}, predicted {}".format(i, true_lables[i], predict_labels[i]))
        return
    def show_topk_result(self, topk_res, true_lables):
        true_lables = ExploreData().get_label_names(true_lables)
        values = topk_res[0]
        indices = topk_res[1]
        img_num = topk_res[0].shape[0]
        for i in range(img_num):
            logging.debug("image {} :True {}, predicted indices {}, predicted values,{}".format(i, true_lables[i], 
                                                                                                ExploreData().get_label_names(indices[i]), values[i]))
        return

   
    def run_graph(self):
        logging.debug("load model ...")
        with tf.Session(graph=self.graph) as sess:
            tf.initialize_all_variables().run()
            model_dir = '../models/'
            if  not os.path.exists(model_dir):
                model_dir = './models/'
            self.restoreModel(sess, model_dir)
            webimg, webimgylabel=self.load_images()
            webimg = self.normalize_images(webimg)
            feed_dict = {self.x_placeholder: webimg, 
                         self.y_true_placeholder: webimgylabel, 
                         self.keep_prob_placeholder: 1.0, 
                         self.phase_train_placeholder:False}
            true_lables = np.argmax(webimgylabel, axis =1)
            if not self.do_topk5:
                predicted_label, acc = sess.run([self.predictedlabel, self.accuracy], feed_dict=feed_dict)
                
                logging.debug("test accuracy {}, predicted labelo{}".format(acc, predicted_label))
                self.show_predict_result(true_lables, predicted_label)
            else:
                topk_res= sess.run(self.topk5, feed_dict=feed_dict)
                self.show_topk_result(topk_res, true_lables)
            
            
            
            
            
            
        return


if __name__ == "__main__":   
    obj= TrafficSignModel_Predict()
    obj.run()