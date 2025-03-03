import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split #for spliting data
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D


path = 'data/myData'
test_ratio = 0.2
validation_ratio = 0.2
image_dimensions = (32,32,3)

#####################################################################

myList = os.listdir(path)
print(myList)
noofdigits = len(myList)
images=[]
digitno = []  # like this is used as lable in future

print('Importing folder of images from 0-9 : ',end='s')
for folder in range(noofdigits):
    myPicList = os.listdir(path+'/'+str(folder))
    for image in myPicList:
        curImg = cv2.imread(path+'/'+str(folder)+'/'+image)  #like ex . data/myData/0/img001-00001.png
        curImg = cv2.resize(curImg,(32,32))
        images.append(curImg)
        digitno.append(folder)
    print(folder,end=" ")
print('\nTotal no of images samples from 0-9 : ',len(images))

#convert it into numpy array
images = np.array(images)
classNo = np.array(digitno)

print('shape of array images : ',images.shape)
#It giving like (10160,32,32,3)  (Total lenght , weidth , height , channels(B,G,R) 3 colored )
print('shape of array class : ',classNo.shape)

###############################################################

#Data spliting using sklearn
x_train,x_test,y_train,y_test = train_test_split(images,classNo,test_size=test_ratio)
print('\nshape of train array after split : ',x_train.shape)
print('shape of test array after split : ',x_test.shape)

#validation spliting
x_train,x_validation,y_train,y_validation = train_test_split(x_train,y_train,test_size=validation_ratio)
print('\nshape of train array after validation : ',x_train.shape)
print('shape of validation array after validation : ',x_validation.shape)

################################################################

flag = 0
if flag == 1 :
    #finding the distribution of 0-9 digit samples
    numOfSamples = []
    for x in range(noofdigits):
        numOfSamples.append(len(np.where(y_train == x)[0]))

    print(numOfSamples)

    #creating barchar for this so that we know distribution
    plt.figure(figsize=(10,5))
    plt.bar(range(0,noofdigits),numOfSamples)
    plt.title('No of images Distribution')
    plt.xlabel('class 10')
    plt.ylabel('Number of Images')
    plt.show()

################################################################

#preprocessing image blur threshold coloring etc
def preprocessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    #normalize image ie its values are from 0-255 to 0-1
    img = img/255
    return img

flag=0#"""Testing for 1 img
if flag==1:
    img = cv2.resize(x_train[30],(300,300))
    cv2.imshow('before_preprocess',img)
    print(img.shape) #colured so channel are 3
    img = preprocessing(x_train[30])
    img = cv2.resize(img,(300,300))
    cv2.imshow('preprocess_img',img)
    print(img.shape)  #because of grayscale it get changed from 3 to 1
    cv2.waitKey(0)

x_train = np.array(list(map(preprocessing,x_train)))
x_test = np.array(list(map(preprocessing,x_test)))
x_validation = np.array(list(map(preprocessing,x_validation)))

#before reshape its like (6502,32,32)
x_train =x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
#after reshape its like (6502,32,32,1)
x_test =x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
x_validation =x_validation.reshape(x_validation.shape[0],x_validation.shape[1],x_validation.shape[2],1)

dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,#zoom in 20% and zoom out 20%
                             rotation_range=10,
                             shear_range=0.1)

dataGen.fit(x_train)

###########################################################
#one hot encoding

y_train = to_categorical(y_train,noofdigits)
y_test = to_categorical(y_test,noofdigits)
y_validation = to_categorical(y_validation,noofdigits)

def myModel():
    noofFilters = 60
    sizeofFilter1 = (5,5)
    sizeofFilter2 = (3,3)
    sizeofPool = (2,2)
    noofNode = 500
    model = Sequential()
    model.add((Conv2D(noofFilters,sizeofFilter1,input_shape=(image_dimensions[0],
                                                             image_dimensions[1]
                                                             ,1),
                      activation='relu',)))
    model.add((Conv2D(noofFilters,sizeofFilter1,activation='relu',)))
    model.add(MaxPooling2D(pool_size=sizeofPool))
    model.add((Conv2D(noofFilters // 2, sizeofFilter2, activation='relu', )))
    model.add((Conv2D(noofFilters // 2, sizeofFilter2, activation='relu', )))
    model.add(MaxPooling2D(pool_size=sizeofPool))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(noofNode,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noofdigits, activation='softmax'))

    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    return model

model = myModel()
print(model.summary())

batchSizeVal=50
epochsVal = 10
stepsperEpoch = 2000

model.fit_generator(dataGen.flow(x_train,y_train,batch_size=batchSizeVal),
                    steps_per_epoch=stepsperEpoch,
                    epochs=epochsVal,
                    validation_data=(x_validation,y_validation),
                    shuffle=1)

#Save model--------
model.save('model/digit_recognisor.h5')