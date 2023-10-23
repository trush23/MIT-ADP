#!/usr/bin/env python
# coding: utf-8

# # **Malaria Detection**

# ##<b>Problem Definition</b>
# **The context:** Why is this problem important to solve?<br>
# 
# Malaria is a long-standing problem, especially in places with tropical weather and poor sanitation. Early detection and treatment is cruicial in reducing the spread and mortality rate from the infection. Until now, there was no option but to rely on off-site laboratory based manual inspection and diagnosis of histology images. The amount of time and effort spent on this tedious process could be better utilized in other areas if this process could be automated. Along with traditional digital image processing, Deep Learning promises to reliably automate this process. It is important to refine ML and DL models to get as accurate a process as we can.
# 
# **The objectives:** What is the intended goal?<br>
# 
# Intended goal here is to create and test various model architetures of Convolutional Neural Networks and assess which one has the best accuracy on test data. We need to look at not just the raw accuracy numbers but also the Confusion Matrix in order to select the best model.
# 
# **The key questions:** What are the key questions that need to be answered?<br>
# 
# 1) What kind of CNN architecture yields the best, most reliable outcome?<br>
# 2) What are the elements that go into a model and what fine-tuning is required in each model? <br>
# 3) What is the effect of preprocessing the data? <br>
# 4) Are the results and metrics of a model in line with what is normally expected of a CNN architecture? <br>
# 
# **The problem formulation:** What is it that we are trying to solve using data science? <br>
# 
# We are trying to analyze histopathology images of uninfected and infected images and create a workflow that automatically identifies infected images as accurately and reliably as possible. For this, we are using supervised learning with labeled data set and convolutional neural network architecture. The problem eventually is of Binary Classification type.
# 
# 
# ## <b>Data Description </b>
# 
# There are a total of 24,958 train and 2,600 test images (colored) that we have taken from microscopic images. These images are of the following categories:<br>
# 
# 
# **Parasitized:** The parasitized cells contain the Plasmodium parasite which causes malaria<br>
# **Uninfected:** The uninfected cells are free of the Plasmodium parasites<br>
# 

# ###<b> Mount the Drive

# In[1]:


# Mounting the drive
from google.colab import drive
drive.mount('/content/drive')


# ### <b>Loading libraries</b>

# In[2]:


# Importing libraries required to load the data
import zipfile
import os
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pandas as pd
import cv2
import tensorflow as tf

# Keras Sequential Model
from tensorflow.keras.models import Model, Sequential
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import load_img, img_to_array
# Importing all the different layers and optimizers. Getting a few extra libraries that may or may not be needed.
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.layers import GlobalAveragePooling2D
# Optimizers
from tensorflow.keras.optimizers import Adam,SGD

# For Categorical variables
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import MinMaxScaler

# To ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Remove the limit from the number of displayed columns and rows. It helps to see the entire dataframe while printing it
pd.set_option("display.max_columns", None)

pd.set_option("display.max_rows", 200)


# ### <b>Let us load the data</b>

# **Note:**
# - You must download the dataset from the link provided on Olympus and upload the same to your Google Drive. Then unzip the folder.

# In[3]:


# Storing the path of the data file from the Google drive
path = '/content/drive/My Drive/Colab Notebooks/capstone/cell_images.zip'

# The data is provided as a zip file so we need to extract the files from the zip file
with zipfile.ZipFile(path, 'r') as zip_ref:

    zip_ref.extractall()


# The extracted folder has different folders for train and test data will contain the different sizes of images for parasitized and uninfected cells within the respective folder name.
# 
# The size of all images must be the same and should be converted to 4D arrays so that they can be used as an input for the convolutional neural network. Also, we need to create the labels for both types of images to be able to train and test the model.
# 
# Let's do the same for the training data first and then we will use the same code for the test data as well.

# In[4]:


# Storing the path of the extracted "train" folder
train_dir = '/content/cell_images/train'

# Size of image so that each image has the same size
SIZE = 64

# Empty list to store the training images after they are converted to NumPy arrays
train_images = []

# Empty list to store the training labels (0 - uninfected, 1 - parasitized)
train_labels = []


# In[5]:


# We will run the same code for "parasitized" as well as "uninfected" folders within the "train" folder
for folder_name in ['/parasitized/', '/uninfected/']:

    # Path of the folder
    images_path = os.listdir(train_dir + folder_name)


    for i, image_name in enumerate(images_path):

        try:

            # Opening each image using the path of that image
            image = Image.open(train_dir + folder_name + image_name)

            # Resizing each image to (64, 64)
            image = image.resize((SIZE, SIZE))

            # Converting images to arrays and appending that array to the empty list defined above
            train_images.append(np.array(image))

            # Creating labels for parasitized and uninfected images
            if folder_name == '/parasitized/':

                train_labels.append(1)

            else:

                train_labels.append(0)

        except Exception:

            pass

# Converting lists to arrays
train_images = np.array(train_images)

train_labels = np.array(train_labels)


# In[6]:


# Storing the path of the extracted "test" folder
test_dir = '/content/cell_images/test'

# Size of image so that each image has the same size (it must be same as the train image size)
SIZE = 64

# Empty list to store the testing images after they are converted to NumPy arrays
test_images = []

# Empty list to store the testing labels (0 - uninfected, 1 - parasitized)
test_labels = []


# In[7]:


# We will run the same code for "parasitized" as well as "uninfected" folders within the "test" folder
for folder_name in ['/parasitized/', '/uninfected/']:

    # Path of the folder
    images_path = os.listdir(test_dir + folder_name)

    for i, image_name in enumerate(images_path):

        try:
            # Opening each image using the path of that image
            image = Image.open(test_dir + folder_name + image_name)

            # Resizing each image to (64, 64)
            image = image.resize((SIZE, SIZE))

            # Converting images to arrays and appending that array to the empty list defined above
            test_images.append(np.array(image))

            # Creating labels for parasitized and uninfected images
            if folder_name == '/parasitized/':

                test_labels.append(1)

            else:

                test_labels.append(0)

        except Exception:

            pass

# Converting lists to arrays
test_images = np.array(test_images)

test_labels = np.array(test_labels)


# ###<b> Check the shape of train and test images

# In[8]:


np.shape(train_images), np.shape(test_images)


# ###<b> Check the shape of train and test labels

# In[ ]:


np.shape(train_labels), np.shape(test_labels)


# ####<b> Observations and insights: There are 24958 training images, each is 64x64x3 and 2600 test images of the same size. All of the data is labeled as well.
# 

# ### <b>Check the minimum and maximum range of pixel values for train and test images

# In[8]:


np.max(train_images), np.min(train_images), np.max(test_images), np.min(test_images)


# ####<b> Observations and insights: The maximum and minimum intensity values in any channel for all images is 255 and 0, respectively.
# 
# 

# ###<b> Count the number of values in both uninfected and parasitized

# In[ ]:


len(np.unique(train_images)), len(np.unique(test_images))


# ###<b>Normalize the images

# In[8]:


# Try to normalize the train and test images by dividing it by 255 and convert them to float32 using astype function
train_images= train_images.astype('float32')/255.0

test_images= test_images.astype('float32')/255.0

train_images_n=cv2.normalize(train_images,None, 0,1,cv2.NORM_MINMAX)
test_images_n=cv2.normalize(test_images,None, 0,1,cv2.NORM_MINMAX)


# In[9]:


np.max(train_images), np.min(train_images), np.max(test_images), np.min(test_images)


# **Observations and Insights: As expected, the maximum and minimum values of Training and Test data is 1.0 and 0.0, respectively.**

# # **Plot to check if the data is balanced (Pandas Dataframe way)**

# In[ ]:


#Trying a Pandas DataFrame approach for barplot

train_uninfected_num=len(os.listdir(train_dir + '/uninfected'))
train_paratisized_num=len(os.listdir(train_dir + '/parasitized'))
test_uninfected_num=len(os.listdir(test_dir + '/uninfected'))
test_paratisized_num=len(os.listdir(test_dir + '/parasitized'))
print("Number of images in each class:   ", train_uninfected_num, train_paratisized_num, test_uninfected_num, test_paratisized_num)


# In[ ]:


plt.figure(figsize = (5, 4))

train_data = {'Uninfected': train_uninfected_num, 'Paraticized': train_paratisized_num}
df_train = pd.Series(train_data)
test_data = {'Uninfected': test_uninfected_num, 'Paraticized': test_paratisized_num}
df_test = pd.Series(test_data)

plt.bar(range(len(df_train)), df_train.values, width=0.3, align = 'center')

plt.bar(range(len(df_test)), df_test.values, width=0.3, align = 'center')

plt.title('Class distribution in training dataset')
plt.xlabel('Class label')
plt.ylabel('Number of samples')
plt.xticks(range(len(df_train)), df_train.index.values, size = 'small')
plt.legend(train_data)
plt.show()


# ###<b> Plot to check if the data is balanced (Matplotlib way)

# In[ ]:


plt.figure(figsize = (5, 4))

# Here it is done via finding Unique Labels.
unique_labels_train, counts_train = np.unique(train_labels, return_counts=True)
unique_labels_test, counts_test = np.unique(test_labels, return_counts=True)
#print('unique labels',unique_labels)
xlabels=['Uninfected','Paratisized']
plt.bar(xlabels, counts_train,width=0.3, align='center')
plt.bar(xlabels, counts_test,width=0.3, align='center')
plt.xlabel('Class label')
plt.ylabel('Number of samples')
plt.title('Class distribution in training dataset')
plt.legend(xlabels)
plt.show()


# ####<b> Observations and insights: The barplots in both ways show that for both categories the counts are roughly equal. Therefore, the data is balanced.

# ### <b>Data Exploration</b>
# Let's visualize the images from the train data

# In[11]:


# This code will help you in visualizing both the parasitized and uninfected images
np.random.seed(42)

plt.figure(1, figsize = (8 , 8))

for n in range(1, 17):

    plt.subplot(4, 4, n)

    index = int(np.random.randint(0, train_images.shape[0], 1))

    if train_labels[index] == 1:

        plt.title('parasitized')

    else:
        plt.title('uninfected')

    plt.imshow(train_images[index])

    plt.axis('off')


# ####<b> Observations and insights: Uninfected images have a uniform intensity in the interior of the cell, whereas infected cells have one or more colored blobs inside the cell body. This is a distinguishing feature that must be picked up by the CNN models accurately. These parasitic blobs are irregular in shape and size and are located at different locations.

# ###<b> Visualize the images with subplot(6, 6) and figsize = (12, 12)

# In[12]:


np.random.seed(42)

plt.figure(1, figsize = (12 , 12))

for n in range(1, 13):

    plt.subplot(6, 6, n)

    index = int(np.random.randint(0, train_images.shape[0], 1))

    if train_labels[index] == 1:

        plt.title('parasitized')

    else:
        plt.title('uninfected')

    plt.imshow(train_images[index])

    plt.axis('off')



# ####<b>Observations and insights: Same as above. We see more variety of cell shapes and intances of infection.

# ###<b> Plotting the mean images for parasitized and uninfected

# In[13]:


#First, define a function to calculate the mean of images.
# "img_mat" is the matrix of the image, "case" is uninfected or paratisized
def mean_img(img_mat, case):

    # Calculate the average
    mean_img = np.mean(img_mat, axis = 0)[0]

    # Reshape it back to a matrix
    plt.imshow(mean_img)

    plt.title(f'Average {case}')

    plt.axis('off')

    plt.show()

    return mean_img


# <b> Mean image for parasitized

# In[14]:


# If the label = 1 then the image is parasitised and if the label = 0 then the image is uninfected
parasitized_data = []  # Create a list to store the parasitized data

for img, label in zip(train_images, train_labels):

        if label == 1:

              parasitized_data.append([img])

parasitized_mean = mean_img(np.array(parasitized_data), 'Parasitized')   # find the mean


# <b> Mean image for uninfected

# In[15]:


# If the label = 1 then the image is parasitised and if the label = 0 then the image is uninfected
uninfected_data = []  # Create a list to store the parasitized data

for img, label in zip(train_images, train_labels):

        if label == 0:

              uninfected_data.append([img])

uninfected_mean = mean_img(np.array(uninfected_data), 'Uninfected')   # find the mean




# ####<b> Observations and insights: They look the same. This is because for parasticized images, the colored blob indicating infection is always at different locations. For each such colored blob, there are many more images where there is no colored blob in that location. Thus, on average, it will show up as a diminished but uniform intenisty patch.

# ### <b>Converting RGB to HSV of Images using OpenCV
# 

# ###<b> Converting the train data

# In[10]:


# The best way to do that is to use OpenCV
#import cv2

hsvarray_train=[]   # to hold the HSV image array

for i in np.arange(0, 100, 1):

  hvtr = cv2.cvtColor(train_images[i], cv2.COLOR_BGR2HSV)

  hsvarray_train.append(hvtr)

hsvarray_train = np.array(hsvarray_train)


# In[11]:


viewimage = np.random.randint(1, 100, 5)

fig, ax = plt.subplots(1, 5, figsize = (10, 10))

for t, i in zip(range(5), viewimage):

  Title = train_labels[i]

  ax[t].set_title(Title)

  ax[t].imshow(hsvarray_train[i].astype(np.uint8))

  ax[t].set_axis_off()

  fig.tight_layout()


# ###<b> Converting the test data

# In[12]:


hsvarray_test=[]   # to hold the HSV image array

for i in np.arange(0, 100, 1):

  hvtst = cv2.cvtColor(test_images[i], cv2.COLOR_BGR2HSV)

  hsvarray_test.append(hvtst)

hsvarray_test = np.array(hsvarray_test)


# In[13]:


viewimage = np.random.randint(1, 100, 5)

fig, ax = plt.subplots(1, 5, figsize = (10, 10))

for t, i in zip(range(5), viewimage):

  Title = test_labels[i]

  ax[t].set_title(Title)

  ax[t].imshow(hsvarray_test[i].astype(np.uint8))

  ax[t].set_axis_off()

  fig.tight_layout()


# ####<b>Observations and insights: All infected cells have blobs inside the cell body.

# ###<b> Processing Images using Gaussian Blurring

# ###<b> Gaussian Blurring on train data

# In[14]:


gbtrain = []  # To hold the blurred images

for i in np.arange(0, 100, 1):

  b = cv2.GaussianBlur(train_images[i], (5, 5), 0)

  gbtrain.append(b)

gbtrain = np.array(gbtrain)


# In[15]:


viewimage = np.random.randint(1, 100, 5)

fig, ax = plt.subplots(1, 5, figsize = (10, 10))

for t, i in zip(range(5), viewimage):

  Title = train_labels[i]

  ax[t].set_title(Title)


  ax[t].imshow(gbtrain[i])

  ax[t].set_axis_off()

  fig.tight_layout()


# ###<b> Gaussian Blurring on test data

# In[16]:


gbtest = []  # To hold the blurred images

for i in np.arange(0, 100, 1):

  b = cv2.GaussianBlur(test_images[i], (5, 5), 0)

  gbtest.append(b)

gbtest = np.array(gbtest)


# In[17]:


viewimage = np.random.randint(1, 100, 5)

fig, ax = plt.subplots(1, 5, figsize = (10, 10))

for t, i in zip(range(5), viewimage):

  Title = test_labels[i]

  ax[t].set_title(Title)


  ax[t].imshow(gbtest[i])

  ax[t].set_axis_off()

  fig.tight_layout()


# ####**Observations and insights: The mostt important observation here is the smoothing of features and removal of noise.**
# 
# **Think About It:** Would blurring help us for this problem statement in any way? What else can we try?
# 
# **Blurring helps with removing noise and hence can benefit overfitting issue later on. The models should not be fitting the noise as well as the features, so the less high frequency noise we have the better. On the other hand, it dilutes the distinguishing feature of paraticized cells, viz. the colored blobs inside the cell body.**
# 
# Other additional operations that will yield very nice identifiable bright spots that are signatures of pathogen in a color-inverted image:
# 
# - Contrast Enhance (possibly CLAHE)
# - Histogram Equalization
# - We could use "open" and "close" morphological operations that do erosion and dilation in sequence to enhance the small blobs in paracitized cells.
# 

# **I will come back to these experiemnts once I am done with all the other questions.**

# **One Hot Encoding the train and test labels**

# In[18]:


# Encoding Train Labels
train_labels = to_categorical(train_labels, 2)

# Encode test labels
test_labels = to_categorical(test_labels, 2)


# ## **Model Building**

# ### **Base Model**
# 
# **Note:** The Base Model has been fully built and evaluated with all outputs shown to give an idea about the process of the creation and evaluation of the performance of a CNN architecture. A similar process can be followed in iterating to build better-performing CNN architectures.

# ###<b> Importing the required libraries for building and training our Model

# In[19]:


from tensorflow.keras import backend
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from random import shuffle


# In[20]:


backend.clear_session()


# In[21]:


np.random.seed(42)

import random

random.seed(42)

tf.random.set_seed(42)


# ###<b> Building the model
# Convolutional Model with the following architecture:<br>
# Three 32 filter layers, kernel size 2, ReLU activation, Max Pooling, and Dropout layers after each convolutional layer. The final layer is a fully connected dense 512 units layer with ReLU activation and a higher dropout that is fed to the dense classifier layer with 2 outputs and a softmax activation function.
# 

# In[22]:


from keras.layers import Dense, LeakyReLU


# In[23]:


model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = 2, padding = "same", activation = "relu", input_shape = (64, 64, 3)))

model.add(MaxPooling2D(pool_size = 2))

model.add(Dropout(0.2))

model.add(Conv2D(filters = 32, kernel_size = 2, padding = "same", activation = "relu"))

model.add(MaxPooling2D(pool_size = 2))

model.add(Dropout(0.2))

model.add(Conv2D(filters = 32, kernel_size = 2, padding = "same", activation = "relu"))

model.add(MaxPooling2D(pool_size = 2))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(512, activation = "relu"))

model.add(Dropout(0.4))

model.add(Dense(2, activation = "softmax")) # 2 represents output layer neurons

model.summary()


# **Compiling the Model**

# In[24]:


model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


# <b> Using Callbacks

# In[25]:


callbacks = [EarlyStopping(monitor = 'val_loss', patience = 3),
             ModelCheckpoint('.mdl_wts.hdf5', monitor = 'val_loss', save_best_only = True)]


# <b> Fit and train our Model

# In[26]:


# Fit the model with min batch size as 32 can tune batch size to some factor of 2^power ]
history = model.fit(train_images, train_labels, batch_size = 32, callbacks = callbacks, validation_split = 0.2, epochs = 30, verbose = 1)


# ###<b> Evaluating the model on test data

# In[27]:


accuracy = model.evaluate(test_images, test_labels, verbose = 1)
print('\n', 'Test_Accuracy:-', accuracy[1])


# <b> Plotting the confusion matrix

# In[28]:


from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

pred = model.predict(test_images)

pred = np.argmax(pred, axis = 1)

y_true = np.argmax(test_labels, axis = 1)

# Printing the classification report
print(classification_report(y_true, pred))

# Plotting the heatmap using confusion matrix
cm = confusion_matrix(y_true, pred)

plt.figure(figsize = (5, 4))

sns.heatmap(cm, annot = True,  fmt = '.0f', xticklabels = ['Uninfected', 'Parasitized'], yticklabels = ['Uninfected', 'Parasitized'])

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.show()


# <b>Plotting the train and validation curves

# In[29]:


# Function to plot train and validation accuracy
def plot_accuracy(history):

    N = len(history.history["accuracy"])

    plt.figure(figsize = (5, 5))

    plt.plot(np.arange(0, N), history.history["accuracy"], label = "train_accuracy", ls = '--')

    plt.plot(np.arange(0, N), history.history["val_accuracy"], label = "val_accuracy", ls = '--')

    plt.title("Accuracy vs Epoch")

    plt.xlabel("Epochs")

    plt.ylabel("Accuracy")

    plt.legend(loc="lower right")


# In[30]:


plot_accuracy(history)


# **Observations: Validation accuracy slightly higher than training accuracy.<br> Precision and Recall numbers are reasonably good. The validation accuracy is dropping so it could have been stopped a bit earlier.**

# So now let's try to build another model with few more add on layers and try to check if we can try to improve the model. Therefore try to build a model by adding few layers if required and altering the activation functions.

# ###<b> Model 1
# ####<b> Trying to improve the performance of our model by adding new layers. Five layers instead of three.
# 

# In[37]:


backend.clear_session()


# In[38]:


model1 = Sequential()

model1.add(Conv2D(filters = 32, kernel_size = 2, padding = "same", activation = "relu", input_shape = (64, 64, 3)))

model1.add(MaxPooling2D(pool_size = 2))

model1.add(Dropout(0.2))

model1.add(Conv2D(filters = 32, kernel_size = 2, padding = "same", activation = "relu"))

model1.add(MaxPooling2D(pool_size = 2))

model1.add(Dropout(0.2))

model1.add(Conv2D(filters = 32, kernel_size = 2, padding = "same", activation = "relu"))

model1.add(MaxPooling2D(pool_size = 2))

model1.add(Dropout(0.2))

model1.add(Conv2D(filters = 32, kernel_size = 2, padding = "same", activation = "relu"))

model1.add(MaxPooling2D(pool_size = 2))

model1.add(Dropout(0.2))

model1.add(Conv2D(filters = 32, kernel_size = 2, padding = "same", activation = "relu"))

model1.add(MaxPooling2D(pool_size = 2))

model1.add(Dropout(0.2))

model1.add(Flatten())

model1.add(Dense(512, activation = "relu"))

model1.add(Dropout(0.4))

model1.add(Dense(2, activation = "softmax")) # 2 represents output layer neurons

model1.summary()


# ###<b> Compiling the model

# In[39]:


model1.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


# <b> Using Callbacks: Trying longer patience=5

# In[40]:


callbacks = [EarlyStopping(monitor = 'val_loss', patience = 5),
             ModelCheckpoint('.mdl_wts.hdf5', monitor = 'val_loss', save_best_only = True)]


# <b>Fit and Train the model

# In[41]:


history1 = model1.fit(train_images, train_labels, batch_size = 32, callbacks = callbacks, validation_split = 0.2, epochs = 20, verbose = 1)


# ###<b> Evaluating the model

# In[42]:


accuracy = model1.evaluate(test_images, test_labels, verbose = 1)
print('\n', 'Test_Accuracy:-', accuracy[1])


# <b> Plotting the confusion matrix

# In[43]:


from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

pred = model1.predict(test_images)

pred = np.argmax(pred, axis = 1)

y_true = np.argmax(test_labels, axis = 1)

# Printing the classification report
print(classification_report(y_true, pred))

# Plotting the heatmap using confusion matrix
cm = confusion_matrix(y_true, pred)

plt.figure(figsize = (5, 4))

sns.heatmap(cm, annot = True,  fmt = '.0f', xticklabels = ['Uninfected', 'Parasitized'], yticklabels = ['Uninfected', 'Parasitized'])

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.show()


# <b> Plotting the train and the validation curves

# In[44]:


# Function to plot train and validation accuracy
def plot_accuracy(history):

    N = len(history.history["accuracy"])

    plt.figure(figsize = (5, 5))

    plt.plot(np.arange(0, N), history.history["accuracy"], label = "train_accuracy", ls = '--')

    plt.plot(np.arange(0, N), history.history["val_accuracy"], label = "val_accuracy", ls = '--')

    plt.title("Accuracy vs Epoch")

    plt.xlabel("Epochs")

    plt.ylabel("Accuracy")

    plt.legend(loc="lower right")


# In[45]:


plot_accuracy(history1)


# **Observations: This model is better than previous model with both accuracies better aligned.**

# ###<b>Think about it:</b><br>
# Now let's build a model with LeakyRelu as the activation function  
# 
# *  Can the model performance be improved if we change our activation function to LeakyRelu?
# *  Can BatchNormalization improve our model?
# 
# Let us try to build a model using BatchNormalization and using LeakyRelu as our activation function.

# ###<b> Model 2 with Batch Normalization

# In[46]:


backend.clear_session() # Clearing the backend for new model


# ###<b> Building the Model.

# In[47]:


model2 = Sequential()

model2.add(Conv2D(filters = 32, kernel_size = 3, padding = "same", input_shape = (64, 64, 3)))
model2.add(LeakyReLU(alpha=0.1))
model2.add(MaxPooling2D(pool_size = 2))
model2.add(BatchNormalization())

#model2.add(Dropout(0.2))

model2.add(Conv2D(filters = 32, kernel_size = 3, padding = "same",input_shape = (64, 64, 3)))
model2.add(LeakyReLU(alpha=0.1))
model2.add(MaxPooling2D(pool_size = 2))
model2.add(BatchNormalization())

#model2.add(Dropout(0.2))

model2.add(Conv2D(filters = 32, kernel_size = 3, padding = "same",input_shape = (64, 64, 3)))
model2.add(LeakyReLU(alpha=0.1))
model2.add(MaxPooling2D(pool_size = 2))
model2.add(BatchNormalization())

#model2.add(Dropout(0.2))

model2.add(Conv2D(filters = 32, kernel_size = 3, padding = "same",input_shape = (64, 64, 3)))
model2.add(LeakyReLU(alpha=0.1))
model2.add(MaxPooling2D(pool_size = 2))
model2.add(BatchNormalization())

#model2.add(Dropout(0.2))

model2.add(Conv2D(filters = 32, kernel_size = 3, padding = "same",input_shape = (64, 64, 3)))
model2.add(LeakyReLU(alpha=0.1))
model2.add(MaxPooling2D(pool_size = 2))
model2.add(BatchNormalization())

#model2.add(Dropout(0.2))

model2.add(Flatten())

model2.add(Dense(512))
model2.add(LeakyReLU(alpha=0.1))

model2.add(Dropout(0.4))

model2.add(Dense(2, activation = "softmax")) # 2 represents output layer neurons

opt = Adam(learning_rate=0.001)

model2.summary()


# ###<b>Compiling the model

# In[48]:


model2.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])


# <b> Using callbacks

# In[49]:


callbacks = [EarlyStopping(monitor = 'val_loss', patience = 5),
             ModelCheckpoint('.mdl_wts.hdf5', monitor = 'val_loss', save_best_only = True)]


# <b>Fit and train the model: <br>
# After running the previous models with a few variations, I found that callbacks is not necessarily the best option and it is good to see what happens to the training and validation accuracy for more epochs, so I will not be using callbacks from now on.

# In[50]:


history2 = model2.fit(train_images, train_labels, batch_size = 32, validation_split = 0.2, epochs = 30, verbose = 1)


# <b>Plotting the train and validation accuracy

# In[51]:


# Function to plot train and validation accuracy
def plot_accuracy(history):

    N = len(history.history["accuracy"])

    plt.figure(figsize = (5, 5))

    plt.plot(np.arange(0, N), history.history["accuracy"], label = "train_accuracy", ls = '--')

    plt.plot(np.arange(0, N), history.history["val_accuracy"], label = "val_accuracy", ls = '--')

    plt.title("Accuracy vs Epoch")

    plt.xlabel("Epochs")

    plt.ylabel("Accuracy")

    plt.legend(loc="lower right")


# In[52]:


plot_accuracy(history2)


# **Observations and Insights:**
# 
# **I have deliberately not used "Callbacks" to see how the model performs over many epochs. If there are Dropout layers in a model, they usually take many epochs to stabilize and affect the network so callbacks maynot be a good idea.**
# 
# **Here, Training Accuracy is close to 1, whereas Validation accuracy is around .98, which is a very nice result. Batch Normalization has indeed helped. Secondly, unlike previous models, Validation accuracy is lower than training accuracy, as it normally should be.**
# 
# **I have also read that BN and Dropout used together tend to make the model worse if used in the wrong order (Reference to be cited later). Here, I have used Dropout after all the BN steps, just before the softmax layer. It has resulted in marked improvement from all my previous attempts (not shown here) for this model using both Dropout and BN together in every layer.**

# ###<b>Evaluating the model

# In[53]:


accuracy = model2.evaluate(test_images, test_labels, verbose = 1)
print('\n', 'Test_Accuracy:-', accuracy[1])


# <b> Generate the classification report and confusion matrix

# In[54]:


from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

pred = model2.predict(test_images)

pred = np.argmax(pred, axis = 1)

y_true = np.argmax(test_labels, axis = 1)

# Printing the classification report
print(classification_report(y_true, pred))

# Plotting the heatmap using confusion matrix
cm = confusion_matrix(y_true, pred)

plt.figure(figsize = (5, 4))

sns.heatmap(cm, annot = True,  fmt = '.0f', xticklabels = ['Uninfected', 'Parasitized'], yticklabels = ['Uninfected', 'Parasitized'])

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.show()


# ####<b>Observations and insights: This is so far the best model. All of the changes mentioned earlier have helped to reduce False Negatives. This is important because it is critical that an infected person is not diagnosed as uninfected.

# ###**Think About It :**<br>
# 
# * Can we improve the model with Image Data Augmentation?<br>
# **We can try and see if it helps.*
# * References to image data augmentation can be seen below:
#   *   [Image Augmentation for Computer Vision](https://www.mygreatlearning.com/blog/understanding-data-augmentation/)
#   *   [How to Configure Image Data Augmentation in Keras?](https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/)
# 
# 
# 
# 

# ###<b>Model 3 with Data Augmentation

# In[55]:


backend.clear_session()


# ###<b> Use image data generator

# In[56]:


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size = 0.2, random_state = 42)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Using ImageDataGenerator to generate images
# I have tried many augmentation steps, pretty much every option that was available, but most don't seem to make any difference.
# Here, I am showing a run with a few basic transformations.
train_datagen = ImageDataGenerator(
                                   horizontal_flip = True,
                                   #vertical_flip = True,
                                   #shear_range=0.2,
                                   zoom_range = 0.5,
                                   rotation_range = 30)

val_datagen  = ImageDataGenerator()

# Flowing training images using train_datagen generator
train_generator = train_datagen.flow(x = X_train, y = y_train, batch_size = 64, seed = 42, shuffle = True)


# Flowing validation images using val_datagen generator
val_generator =  val_datagen.flow(x = X_val, y = y_val, batch_size = 64, seed = 42, shuffle = True)


# ###**Think About It :**<br>
# 
# *  Check if the performance of the model can be improved by changing different parameters in the ImageDataGenerator.
# 
# 

# **I checked with every option in ImageDataGenerator like shearing, shifts, flips, etc. I did not see a significant improvement. In fact, most of the time the results were worse.**

# ####<B>Visualizing Augmented images

# In[57]:


# Creating an iterable for images and labels from the training data
images, labels = next(train_generator)

# Plotting 16 images from the training data
fig, axes = plt.subplots(4, 4, figsize = (8, 8))

fig.set_size_inches(8, 8)
for (image, label, ax) in zip(images, labels, axes.flatten()):

    ax.imshow(image)

    if label[1] == 1:

        ax.set_title('parasitized')

    else:

        ax.set_title('uninfected')

    ax.axis('off')


# ####<b>Observations and insights: I am not sure why those long bands connecting the edge of the cell to image boundary is forming. It is somewhat undesirable. I have to investigate various data augmentation possibilities in more detail.

# ###<b>Building the Model

# In[58]:


model3 = Sequential()

model3.add(Conv2D(filters = 32, kernel_size = 3, padding = "same", input_shape = (64, 64, 3)))
model3.add(LeakyReLU(alpha=0.1))
model3.add(MaxPooling2D(pool_size = 2))
model3.add(BatchNormalization())

#model3.add(Dropout(0.2))

model3.add(Conv2D(filters = 32, kernel_size = 3, padding = "same",input_shape = (64, 64, 3)))
model3.add(LeakyReLU(alpha=0.1))
model3.add(MaxPooling2D(pool_size = 2))
model3.add(BatchNormalization())

#model3.add(Dropout(0.2))

model3.add(Conv2D(filters = 32, kernel_size = 3, padding = "same",input_shape = (64, 64, 3)))
model3.add(LeakyReLU(alpha=0.1))
model3.add(MaxPooling2D(pool_size = 2))
model3.add(BatchNormalization())

#model3.add(Dropout(0.2))

model3.add(Conv2D(filters = 32, kernel_size = 3, padding = "same",input_shape = (64, 64, 3)))
model3.add(LeakyReLU(alpha=0.1))
model3.add(MaxPooling2D(pool_size = 2))
model3.add(BatchNormalization())

#model3.add(Dropout(0.2))

model3.add(Conv2D(filters = 32, kernel_size = 3, padding = "same",input_shape = (64, 64, 3)))
model3.add(LeakyReLU(alpha=0.1))
model3.add(MaxPooling2D(pool_size = 2))
model3.add(BatchNormalization())

#model3.add(Dropout(0.2))

model3.add(Flatten())

model3.add(Dense(512))
model3.add(LeakyReLU(alpha=0.1))

model3.add(Dropout(0.4))

model3.add(Dense(2, activation = "softmax")) # 2 represents output layer neurons

opt = Adam(learning_rate=0.001)

model3.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])

model3.summary()


# <b>Using Callbacks

# In[ ]:


callbacks = [EarlyStopping(monitor = 'val_loss', patience = 3),
             ModelCheckpoint('.mdl_wts.hdf5', monitor = 'val_loss', save_best_only = True)]


# <b> Fit and Train the model

# In[59]:


history3 = model3.fit(train_generator,
                                  validation_data = val_generator,
                                  batch_size = 32,
                                  epochs = 30, verbose = 1)


# ###<B>Evaluating the model

# <b>Plot the train and validation accuracy

# In[60]:


plot_accuracy(history3)


# In[61]:


accuracy3 = model3.evaluate(test_images, test_labels, verbose = 1)
print('\n', 'Test_Accuracy:-', accuracy[1])


# <B>Plotting the classification report and confusion matrix

# In[62]:


from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

pred = model3.predict(test_images)

pred = np.argmax(pred, axis = 1)

y_true = np.argmax(test_labels, axis = 1)

# Printing the classification report
print(classification_report(y_true, pred))

# Plotting the heatmap using confusion matrix
cm = confusion_matrix(y_true, pred)

plt.figure(figsize = (5, 4))

sns.heatmap(cm, annot = True,  fmt = '.0f', xticklabels = ['Uninfected', 'Parasitized'], yticklabels = ['Uninfected', 'Parasitized'])

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.show()


# <b> Now, let us try to use a pretrained model like VGG16 and check how it performs on our data.

# ### **Pre-trained model (VGG16)**
# - Importing VGG16 network upto block 5, keeping it non-trainable.
# - Adding Fully Connected Layers on top of it

# In[63]:


# Clearing backend
from tensorflow.keras import backend

backend.clear_session()

# Fixing the seed for random number generators
np.random.seed(42)

import random

random.seed(42)

tf.random.set_seed(42)


# In[64]:


from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras import Model

vgg = VGG16(include_top = False, weights = 'imagenet', input_shape = (64, 64, 3))

vgg.summary()


# In[65]:


transfer_layer = vgg.get_layer('block5_pool')

vgg.trainable = False

# Flatten the output from the 3rd block of the VGG16 model
# Add classification layers on top of it
x = Flatten()(transfer_layer.output)
x = Dense(256, activation = 'relu')(x)
x = BatchNormalization()(x)
#x = Dropout(0.3)(x)

# Similarly add a dense layer with 128 neurons
x = Dense(128, activation = 'relu')(x)
x = BatchNormalization()(x)
#Dropout layer
#x = Dropout(0.3)(x)

# Add a dense layer with 64 neurons
x = Dense(64, activation = 'relu')(x)
x = BatchNormalization()(x)
# Again, keeping dropout after all the BN layers
x = Dropout(0.3)(x)
pred = Dense(2, activation = 'softmax')(x)

model4 = Model(vgg.input, pred) # Initializing the model

model4.summary()


# ###<b>Compiling the model

# In[66]:


opt = Adam(learning_rate=0.001)

model4.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])


# <b> using callbacks

# In[67]:


callbacks = [EarlyStopping(monitor = 'val_loss', patience = 5),
             ModelCheckpoint('.mdl_wts.hdf5', monitor = 'val_loss', save_best_only = True)]


# <b>Fit and Train the model

# In[68]:


history4 = model4.fit(train_generator,
                                  validation_data=val_generator,
                                  validation_split=0.2,
                                  batch_size = 32,
                                  epochs = 30, verbose = 1)


# In[69]:


# Function to plot train and validation accuracy
def plot_accuracy(history):

    N = len(history.history["accuracy"])

    plt.figure(figsize = (5, 5))

    plt.plot(np.arange(0, N), history.history["accuracy"], label = "train_accuracy", ls = '--')

    plt.plot(np.arange(0, N), history.history["val_accuracy"], label = "val_accuracy", ls = '--')

    plt.title("Accuracy vs Epoch")

    plt.xlabel("Epochs")

    plt.ylabel("Accuracy")

    plt.legend(loc="lower right")


# <b>Plot the train and validation accuracy

# In[70]:


plot_accuracy(history4)


# ###**Observations and insights: _____**
# 
# *   What can be observed from the validation and train curves?
# 
# **We are back to validation accuracy being higher than training accuracy. This could be due to not having enough data augmentation. It is possible that if more variations in the data are incorporated, both accuracies maybe closer. It could also be that more fine tuning is required in the final layers. I also read that VGG16 is trained on images normalized by subtracting the mean. Our images are normalized differently and maybe they need to be normalized the same way.**

# ###<b> Evaluating the model

# In[71]:


accuracy4 = model4.evaluate(test_images, test_labels, verbose = 1)
print('\n', 'Test_Accuracy:-', accuracy4[1])


# <b>Plotting the classification report and confusion matrix

# In[72]:


from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

pred = model4.predict(test_images)

pred = np.argmax(pred, axis = 1)

y_true = np.argmax(test_labels, axis = 1)

# Printing the classification report
print(classification_report(y_true, pred))

# Plotting the heatmap using confusion matrix
cm = confusion_matrix(y_true, pred)

plt.figure(figsize = (5, 4))

sns.heatmap(cm, annot = True,  fmt = '.0f', xticklabels = ['Uninfected', 'Parasitized'], yticklabels = ['Uninfected', 'Parasitized'])

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.show()


# ###<b>Think about it:</b>
# *  What observations and insights can be drawn from the confusion matrix and classification report? <br>
# **This is the worst performing model of the lot. That could be because I haven't fully mastered the art of transfer learning and fine tuning transfer learning models.*
# *  Choose the model with the best accuracy scores from all the above models and save it as a final model.
# 
# **Based on both the accuracy numbers and the Confusion Matrix, the second model with five layers and batch normalization was the best. The third model with data augmentation is also close second. The accuracy numbers are very good but confusion matrix shows a bit higher instances of parasitized cases classified as uninfected.**
# 

# ####<b> Observations and Conclusions drawn from the final model: The second model not only had excellent training and validation accuracy, it also had equally good Test Accuracy. Moreover, from Confusion Matrix, it had the least number of False negatives, meaning, it had the lowest number of cases where a parasitized cell was identified as uninfected. This is highly desireable to prevent fatalities and spread of infection.
# 
# 

# **Improvements that can be done:**<br>
# 
# 
# *  Can the model performance be improved using other pre-trained models or different CNN architecture?
# 
# **We can try ResNet50 architecture as a pre-trained model. We can also employ image pre-processing steps like contrast enhancement, histogram equalization, single channel processing etc.**
# 
# *  You can try to build a model using these HSV images and compare them with your other models.
# 
# **I plan to exeriment with not just HSV but also the LAB colormap images. They are particularly suitable for CLAHE and other contrast enhancing techniques.**

# #### **Insights**
# 
# ####**Refined insights**:
# - What are the most meaningful insights from the data relevant to the problem?
# 
# **The most meaningful insight here is that the distinguishing feature of parasitized cells is a small blobs that are darker than the background but otherwise the image has uniform intensity. This mean that the CNN architecture doesn't need to be overly complicated and any reasonably deep network with appropriate batch normalization or dropout or regularization method will work.**
# 
# **It is indeed possible to automate the pocess of identifying infected cells in a relatively short time. It doesn't take too long to run on my machine.**
# 
# ####**Comparison of various techniques and their relative performance**:
# - How do different techniques perform? Which one is performing relatively better? Is there scope to improve the performance further?
# 
# **1) The base model with three Convolutional layers itself was pretty good. The only issue was that Validation accuracy (0.98) was higher than training accuracy (0.97) possibly due to similarities in the validation set and not enough normalization of the distribution of the data. The Test Accuracy was 0.985. Weighted average of both the Precision and recall was 0.99! <br>**
# **2) The second model with more layers was relatively worse than the base model. It had five convolutional layers with dropouts. The training, validation, and Test accuracies matched closely (~0.98) but there was a slight drop in Precision and Recall. Adding layers didn't improve performace by itself.**<br>
# **3) The third model (Model2) with LeakyReLU and Batch Normalization was the best performing model. It had five layes, 32 filters each with max pooling and Batch Normalization. Dropout was only before the classification layer. It had all the desirable features. High Training and Validation accuracy, 0.998 and 0.979, respectively. High Testing Accuracy of 0.983. Precsion and Recall of 0.98 and the lowest cases of parasitized cells identified as uninfected.**<br>
# **4) The Data Augmented model was also very good in terms of Accuracy (all close to 0.98) and Confusion Matrix numbers also close to 0.99. The only issue that I didn't understand was that the trend of validation accuracy being higher than Training accuracy came back. Data Augmentation was supposed to do the opposite!**<br>
# **5) Transfer Learning with VGG16 was the poorest performing model. It could be that the model was not trained on such medical images, but that is highly unlikely with millions of training images. I would not use transfer learning for now until further experiemnets and fine tuning.**
# 
# ####**Proposal for the final solution design**:
# - What model do you propose to be adopted? Why is this the best solution to adopt?<br>
# 
# **I propose to adopt Model2 with five convolutional layers, Maxpooling and Batch Normalization and dropout for the last layer.**

# 
