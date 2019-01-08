'''
purpose of program is to classify an image as containing a cat or a dog
created by - ammanuel selameab, jan 2019
'''

import cv2
import os
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


#process image files
img_data = []
img_labels = []
    
img_path = "C:\\PythonPrograms\\cat_dog_data\\train\\"
img_class_names = os.listdir(img_path)
    
    
labels_name={'cat':0,'dog':1}

for i in img_class_names: #go through each class directory (cat and dog) separately
    img_names = os.listdir(img_path + "\\" + i) #img_names contains the name of every image within the cat folder
    label = labels_name[i]
    print("processing " + i +" images")
    for j in img_names:
        img_arr = cv2.imread(img_path + "\\" + i + "\\" + j)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        img_arr_resize = cv2.resize(img_arr,(128,128))
        img_data.append(img_arr_resize)
        img_labels.append(label)
    
img_data = np.array(img_data)
img_data = img_data.astype('float32')
img_data /= 255 #normalize data set to 0-1 range to ease in processing time
    
img_labels = np.array(img_labels)

num_classes = 2
img_labels_encod = np_utils.to_categorical(img_labels, num_classes) #transform labels via one-hot encoding

x_train, x_test, y_train, y_test = train_test_split(img_data, img_labels_encod, test_size=0.2, random_state=2)

#create a model and add layers

model = Sequential()

model.add(Conv2D(32, (3,3), padding='same', input_shape=(32, 32, 3), activation="relu"))
model.add(Conv2D(32, (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))
    
# print summary of model
model.summary()

# compile the model
model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])

# train the model
#model.fit(x_train,y_train,batch_size=32,epochs=30,validation_data=(x_test, y_test),shuffle=True)
model.fit(x_train, y_train, batch_size=16, epochs=20, verbose=1, validation_data=(x_test, y_test))

# save neural network structure
model_structure = model.to_json()
f = Path("model_structure.json")
f.write_text(model_structure)

# save neural network's trained weights
model.save_weights("model_weights")

