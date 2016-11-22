import numpy as np
import cv2

class ImageAugmentation():
    def __init__(self):
        return
    def translation(self, img, shift_range = 2):
        x = int(np.random.random() * shift_range *2) - shift_range
        y = int(np.random.random() * shift_range *2) - shift_range
        
        rows,cols,_= img.shape
        M = np.float32([[1,0,x],[0,1,y]])
        img_trans = cv2.warpAffine(img, M,(cols,rows))
        return img_trans
    def rotation(self, img, angle=15.0, scale = 0.1):
        degree = int(np.random.random()*angle*2)-angle
        ratio = np.random.random()*scale*2 - scale + 1
        
        rows,cols, _ = img.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),degree,ratio)
        img_trans = cv2.warpAffine(img,M,(cols,rows))
        return img_trans
    
    def __transform_image_1(self, img,ang_range,shear_range,trans_range):
        '''
        This function transforms images to generate new images.
        The function takes in following arguments,
        1- Image
        2- ang_range: Range of angles for rotation
        3- shear_range: Range of values to apply affine transform to
        4- trans_range: Range of values to apply translations over. 
        
        A Random uniform distribution is used to generate different parameters for transformation
        
        '''
        # Rotation
    
        ang_rot = ang_range * np.random.uniform()-ang_range/2
        rows,cols,ch = img.shape    
        Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)
    
        # Translation
        tr_x = trans_range*np.random.uniform()-trans_range/2
        tr_y = trans_range*np.random.uniform()-trans_range/2
        Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    
        # Shear
        pts1 = np.float32([[5,5],[20,5],[5,20]])
    
        pt1 = 5+shear_range*np.random.uniform()-shear_range/2
        pt2 = 20+shear_range*np.random.uniform()-shear_range/2
    
        pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])
    
        shear_M = cv2.getAffineTransform(pts1,pts2)
            
        img = cv2.warpAffine(img,Rot_M,(cols,rows))
        img = cv2.warpAffine(img,Trans_M,(cols,rows))
        img = cv2.warpAffine(img,shear_M,(cols,rows))
        
        return img
    def __transform_image(self, img, shift, angle, scale):
        img = self.translation(img, shift)
        img = self.rotation(img, angle, scale)
        return img
    def transform_image(self, img):
        angle = 15 #rotation angle 15 degree
        scale = 0.1 #
        shift = 2 #both x an y shift 2
        return self.__transform_image(img, shift, angle, scale)
    def transform_imagebatch(self, batch_image):
        N = batch_image.shape[0]
        res =[]
        for n in range(N):
            temp = self.transform_image(batch_image[n])
            res.append(temp)
        return np.array(res)
        
    def run(self):
        img = cv2.imread('../data/sift_basic_0.jpg', cv2.IMREAD_COLOR)
        cv2.imshow('image',img)
        
#         img = self.translation(img,shift_range = 20)
#         img = self.rotation(img, 45, 0.9)
        img = self.transform_image(img)
        cv2.imshow('image_scaled',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    

if __name__ == "__main__":   
    obj= ImageAugmentation()
    obj.run()