#!/usr/bin/env python
# coding: utf-8

# # Question
# Use Aritificial Neural Netwrok(ANN) to accurately predict the digits in the MNIST dataset

# #### MNIST dataset 
# MNIST is a collection of handwritten digits ranging from the number 0 to 9.It has a training set of 60,000 images and 10,000 test images that are classified into corresponding categories or labels.In Keras the MNIST dataset is preloaded in the form of four Numpy arrays.

# #### importing necessary libraries

# In[68]:


import keras
import numpy as np
from keras.datasets import  mnist


# #### spliting dataset into training and test set

# In[69]:


(x_train,y_train),(x_test,y_test)=mnist.load_data()


# In[70]:


#display images
import matplotlib.pyplot as plt
plt.imshow(x_train[7],cmap=plt.cm.binary)


# In[71]:


print(y_train[7])


# In[72]:


print(x_train.shape)


# The data consists of 60,000 images of dimension 28*28

# In[73]:


#view the data type of tensor
x_train.dtype


# #### Data normalization 
# 
# *The MNIST images of 28×28 pixels are represented as an array of numbers whose values range from [0, 255] of type uint8.
# 
# *It is usual to scale the input values of neural networks to certain ranges.
# 
# *In this method, the input values should be scaled to values of type float32 within the interval [0, 1].

# In[74]:


x_train=x_train.astype('float32')
x_test=x_test.astype('float32')


# In[75]:


x_train/=255
x_test/=255


# In[76]:


#reshaping the x from 3 dimension to 2 dimension
x_train=x_train.reshape(60000,784)
x_test=x_test.reshape(10000,784)


# In[77]:


x_train.shape


# #### Encoding y using onehot encoding

# Now we have the labels for each input data.
# They are numbers between 0 and 9 that indicate which digit represents the image, that is, to which class they are associated.
# We will represent this label with a vector of 10 positions, where the position corresponding to the digit that represents the image contains a 1 and the remaining positions of the vector contain the value 0.

# In[78]:


from keras.utils import to_categorical


# In[79]:


y_train.shape


# In[80]:


y_train=to_categorical(y_train,num_classes=10)
y_test=to_categorical(y_test,num_classes=10)


# In[81]:


y_train[2]


# In[82]:


y_train.shape


# #### Defining the model

# In[83]:


from keras import Sequential
from keras.layers.core import Dense,Activation


# In[84]:


model=Sequential()


# #### Adding layers in Artificial Neural Network

# In[85]:


model.add(Dense(10,activation='sigmoid',input_shape=(784,)))
model.add(Dense(10,activation='softmax'))


# In[86]:


model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])


# In[87]:


model.fit(x_train,y_train,batch_size=100,epochs=10)


# #### Model summary

# In[88]:


model.summary()


# In[ ]:


From this model summary we can see that 7850 


# #### Evaluating the model

# In[89]:


test_loss, test_acc=model.evaluate(x_test,y_test)


# #### Accuracy of the model

# In[90]:


print('Test accuracy is',round(test_acc,4))


# Thus the model has 83% accuracy

# #### Plotting confusion matrix

# In[91]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=30)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual class')
    plt.xlabel('Predicted class')


# In[92]:


from collections import Counter
from sklearn.metrics import confusion_matrix
import itertools

# Predict the values from the validation dataset
Y_pred = model.predict(x_test)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred, axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_test, axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10))


# # Interpretation

# We have used simple implementation of a Artificial Neural Network model using Keras library to recognize handwritten digits from the MNIST dataset. The process starts with loading the dataset and displaying an image from the dataset along with its corresponding label. The data is then preprocessed by normalizing the pixel values and reshaping the input data to 2D. The output data is also converted to categorical format.
# 
# A Artificial Neural Network model is then defined using Sequential API from Keras, consisting of two dense layers. The model is compiled using categorical cross-entropy loss and stochastic gradient descent optimizer. The model is then trained on the training dataset for 10 epochs with a batch size of 100.
# 
# After training, the model is evaluated on the test dataset and the test accuracy is printed. Finally, a confusion matrix is plotted using the predicted classes and true labels from the test dataset. The confusion matrix shows how well the model has classified each class and where it has made errors.
# This model have accuracy score of 0.8425 and this is a good ANN for classification.

# # Question

# Use Aritificial Neural Netwrok(ANN) to accurately predict whether the daigonised by cancer using Breast cancer dataset

# In[1]:


# Import libraries
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np


# In[2]:


#Load dataset
data=pd.read_csv("data new.csv")
data.head()


# In[3]:


data.describe()


# In[4]:


data.info()


# In[5]:


data.isnull().sum()


# In[6]:


x = data.iloc[:,2:32]
y = data.iloc[:,1]


# In[7]:


from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)


# In[8]:


#splitting into training and testing set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[9]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[10]:


#importing the model & layer
import keras
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential() # Initialising the ANN

classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 30))
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


# In[11]:


#Compiling ANN
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[12]:


#Fitting the ANN
classifier.fit(x_train, y_train, batch_size = 1, epochs = 100)


# In[13]:


# Model Evaluation
y_pred = classifier.predict(x_test)
y_pred = [ 1 if y>=0.5 else 0 for y in y_pred ]
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

print("accuracy score: ",accuracy_score(y_pred , y_test))
print("confusion matrix: \n",confusion_matrix(y_test,y_pred))


# # Interpretation

# An Artitificial Neural Network has been implimented in the breast cancer dataset. This ANN consists of 4 layers containing 16, 8, 6, and 1 neurons. We have used 3 relu and 1 sigmoid activation function. Here we use RMSprop as optimizer and the binary cross-entropy loss function as loss function for binary classification. Finally, the accuracy metric is used to monitor the performance of the model during training and testing. In the first epoch we have got an acurracy of 0.65 and at the 100th epoch it reaches 0.99. From the confusion matrix we can interpret that 110/114 of the observations are correctly classified and 4/114 are misclassified.This model have accuracy score of 0.96 and this is a good ANN for classification.

# # Question

# Use Aritificial Neural Netwrok(ANN) to accurately predict whether the customer churn using Churn Modelling dataset

# In[16]:


# Import libraries
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np


# In[17]:


# Load dataset
df=pd.read_csv('Churn_Modelling.csv')
df.head()


# In[18]:


df.describe()


# In[24]:


df.info()


# In[25]:


df.isnull().sum()


# In[26]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Geography'] = le.fit_transform(df['Geography'])
df['Gender'] = le.fit_transform(df['Gender'])


# In[27]:


x = df.iloc[:,3:13].values 
y =  df.iloc[:,13] .values


# In[29]:


from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
z = one.fit_transform(x[:,1:2]).toarray()
x = np.delete(x,1,axis = 1)
x = np.concatenate((z,x),axis = 1)


# In[30]:


x.shape


# In[32]:


x  = x[:,1:]


# In[33]:


x.shape


# In[34]:


# Splitting into training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)


# In[35]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[36]:


#importing the model & layer
import keras 
from keras.models import Sequential
from keras.layers import Dense


# In[37]:


classifer = Sequential()
classifer.add(Dense(units = 11,kernel_initializer = 'uniform', activation = 'relu'))
classifer.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifer.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu'))
classifer.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

classifer.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
classifer.fit(x_train,y_train, batch_size = 16, epochs = 200)


# In[38]:


y_pred = classifer.predict(x_test)


# In[39]:


y_pred = (y_pred>0.5)


# In[40]:


# Model Evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

print("accuracy score: ",accuracy_score(y_pred , y_test))
print("confusion matrix: \n",confusion_matrix(y_test,y_pred))


# # Interpretation

# An Artitificial Neural Network has been implimented in the Churn Modelling dataset. We load the dataset using pandas and performs some exploratory data analysis by checking the data with the head, describe, info, and isnull functions. It also performs label encoding on the categorical variables 'Geography' and 'Gender' using LabelEncoder from scikit-learn.\
# The independent variables and dependent variable are then separated and stored in the variables x and y, respectively. OneHotEncoder is applied on the second column of x to encode it into a binary array, and the resulting array is concatenated with the rest of the x columns. The first column of the concatenated x array is then dropped.The x and y variables are split into training and testing sets using the train_test_split function from scikit-learn. The StandardScaler from scikit-learn is used to scale the features.\
# A neural network model is then created using the Sequential class from Keras. The model consists of four layers, with the first three layers using the ReLU activation function and the last layer using the sigmoid activation function. The model is compiled with the binary_crossentropy loss function and the Adam optimizer. The model is then trained on the training data using the fit function.The model's predictions are then made on the testing data and compared with the actual values. The model has an accuracy score of 0.86 this means that the given ANN is a good model.

# In[ ]:




