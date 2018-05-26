
# coding: utf-8

# In[12]:


import cv2 
import os
import glob
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
from pathlib import Path
import matplotlib.image as mpimg
from matplotlib   import pyplot as plt
from math import pi
import pickle
get_ipython().magic('matplotlib inline')


# Test hàm

# In[2]:


## Lấy danh sách label trong tập train


# In[3]:


def get_labels(train_path):
    p=Path(train_path)
    path_file =[] # return absolute path
    labels = []
    if p.is_dir():
        for file in p.iterdir():
            if file.is_dir():
                path_file.append(file)
                
        for path in path_file:
            label = path.parts[-1]
            labels.append(label)
#             print(path.parts[-1])
    else:
        raise ValueError(f"{train_path} is not a directory!!")
    return labels


# a= get_labels(r"D:\DatasetJapanese\data_use_kl\00753.png") // error
# print(a)    

# b= get_labels(r"D:\DatasetJapanese\data_use_kl")
# print((b[0])) #label


# GenerateData

# In[4]:


IMAGE_SIZE=64
def rotate_images(X_imgs, start_angle=45, end_angle=-45, n_images=2):
    X_rotate = []
#     iterate_at = (end_angle - start_angle) / (n_images - 1)
    do = np.random.uniform(end_angle,start_angle,n_images)
    
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (None, IMAGE_SIZE, IMAGE_SIZE, 3))
    radian = tf.placeholder(tf.float32, shape = (len(X_imgs)))
    tf_img = tf.contrib.image.rotate(X, radian)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    
        for index in range(n_images):
            degrees_angle = do[index]
            radian_value = degrees_angle * pi / 180  # Convert to radian
            radian_arr = [radian_value] * len(X_imgs)
            rotated_imgs = sess.run(tf_img, feed_dict = {X: X_imgs, radian: radian_arr})
            X_rotate.extend(rotated_imgs)

    X_rotate = np.array(X_rotate, dtype = np.float32)
    return X_rotate


def central_scale_images(X_imgs, scales=np.round(np.random.uniform(0.8,1,3),2)):
    # Various settings needed for Tensorflow operation
    
    boxes = np.zeros((len(scales), 4), dtype = np.float32)
    for index, scale in enumerate(scales):
        x1 = y1 = 0.5 - 0.5 * scale # To scale centrally
        x2 = y2 = 0.5 + 0.5 * scale
        boxes[index] = np.array([y1, x1, y2, x2], dtype = np.float32)
    box_ind = np.zeros((len(scales)), dtype = np.int32)
    crop_size = np.array([IMAGE_SIZE, IMAGE_SIZE], dtype = np.int32)
    
    X_scale_data = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (1, IMAGE_SIZE, IMAGE_SIZE, 3))
    # Define Tensorflow operation for all scales but only one base image at a time
    tf_img = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for img_data in X_imgs:
            batch_img = np.expand_dims(img_data, axis = 0)
            scaled_imgs = sess.run(tf_img, feed_dict = {X: batch_img})
            X_scale_data.extend(scaled_imgs)
    
    X_scale_data = np.array(X_scale_data, dtype = np.float32)
    return X_scale_data
def generate_data(X_imgs):
    
    rotated_imgs = rotate_images(X_imgs)
    scaled_imgs =central_scale_images(rotated_imgs)
    
    result = np.concatenate((X_imgs,rotated_imgs,scaled_imgs,),axis=0)
    return result
    


# In[5]:


def load_train(train_path,img_size):
    images=[]
    labels=[]
    img_names=[]
    cls = []
    classes = get_labels(train_path)
    
    
    for fields in classes:
        index = classes.index(fields)
        # print("doc file {} (index: {})".format(fields,index))
        path = os.path.join(train_path,fields,'*')
#         print(f"path: {path}")
        files = glob.glob(path) #return list of path names that match path =path;
#         print("file: ",files)
        
        for fi in files:
            img = cv2.imread(fi);
            img= cv2.resize(img,(img_size,img_size))
            img=img.astype(np.float32)
            img=np.multiply(img, 1.0/255.0)
            img = np.reshape(img,(-1,64,64,3))
            data_is_generated = generate_data(img) #shpae [num_img,img_size,img_size,channel]
            filename_base = os.path.basename(fi)
            for i in range (data_is_generated.shape[0]):
                images.append(data_is_generated[0])
                label = np.zeros(len(classes))
                label[index]=1
                labels.append(label)
                img_names.append(filename_base)
                cls.append(fields)
          
  
    images=np.array(images)
    labels=np.array(labels)
    img_names=np.array(img_names)
    cls = np.array(cls)
    
    return images,labels,img_names,cls


# In[6]:


# # test load_train function
# images,labels,img_names,cls= load_train("D:\DatasetJapanese\data_katagana\katakana_test",64)
# print(labels[0],img_names[0],cls[0])
# img = plt.imshow(images[0])
# print(images.shape) # (total_images,img_size,img_size,channels)
# print(images.shape[0])# total_images


# In[7]:


class DataSet (object):
    
    def __init__(self,images,labels,img_names,cls):
        # print("size: {} ".format(images.shape[0]))
        self._num_examples=images.shape[0]
        
        self._images= images
        self._labels=labels
        self._img_names=img_names
        self._cls = cls;
        
        self._epochs_done = 0
        self._index_in_epoch = 0
        
    @property
    def images(self):
        return self._images
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def img_names(self):
        return self._img_names
    
    @property
    def cls(self):
        return self._cls
    
    @property
    def num_examples(self):
        return self._num_examples
    
    @property
    def epochs_done(self):
        return self._epochs_done


    def next_batch(self,batch_size):
        start=self._index_in_epoch
        self._index_in_epoch +=batch_size
        
        if(self._index_in_epoch >self.num_examples):
            # sau moi epoch can phai shuffle lai vi tri de du lieu duoc ngau nhien, tranh lap lai batchsize giong nhau
            self._images,self._labels,self._img_names,self._cls = shuffle(self._images,self._labels,self._img_names,self._cls)
            
            self._epochs_done +=1
            start=0
            self._index_in_epoch=batch_size
            assert batch_size <= self._num_examples
            
        end = self._index_in_epoch
        
        return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]


# In[8]:



def read_train_sets(train_path,image_size,test_size, validation_size):
    classes = get_labels(train_path)
    class DataSets(object):pass
    
    data_sets = DataSets();
    
    images,labels,img_names,cls = load_train(train_path,image_size)
    images,labels,img_names,cls = shuffle(images,labels,img_names,cls)
    
    if isinstance(test_size,float) or isinstance(test_size,float) or test_size<1 or validation_size<1 :
        test_size=int(images.shape[0] * test_size) # total_images * validation_size
        validation_size = test_size + int(images.shape[0] * validation_size)
        # print("images: {}".format(images.shape[0]))
        # print("valid: {}".format(validation_size))
        
    test_images = images[:test_size]
    test_labels = labels[:test_size]
    test_img_names = img_names[:test_size]
    test_cls = cls[:test_size]
    
    validation_images = images[test_size:validation_size]
    validation_labels = labels[test_size:validation_size]
    validation_img_names = img_names[test_size:validation_size]
    validation_cls = cls[test_size:validation_size]
    
    train_images = images[validation_size:]
    train_labels = labels[validation_size:]
    train_img_names = img_names[validation_size:]
    train_cls = cls[validation_size:]
    
    data_sets.test = DataSet(test_images,test_labels,test_img_names,test_cls)
    data_sets.train= DataSet(train_images,train_labels,train_img_names,train_cls)
    data_sets.valid = DataSet(validation_images,validation_labels,validation_img_names,validation_cls)
    
    return data_sets




# validation_size = 0.22
# train_path='D:\DatasetJapanese\\data_katagana\katakana_test'

# data=read_train_sets("D:\DatasetJapanese\data_katagana\katakana_test",image_size=10,validation_size=validation_size)
# images,labels,img_names,cls = load_train(train_path,64)
# print(data.train.images.shape)
# print(data.valid.images.shape)
# print(images.shape)


# In[13]:


# data = read_train_sets(r"D:\DatasetJapanese\data_katagana\minitest",64,0.2,0.1)
# class DataCompression(object):
#     def __init__(self,data):
#         self.train = data.train
#         self.valid = data.valid
#         self.test = data.test
        
# data_train = DataCompression(data)
# with open("train.file", "wb") as f:
#     pickle.dump(data_train, f, pickle.HIGHEST_PROTOCOL)


# In[17]:


# data.train.images.shape


# In[20]:


# np.random.uniform(10,-10,5)

