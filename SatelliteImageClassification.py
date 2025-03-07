
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import os
import re
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import cv2
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle


main = tkinter.Tk()
main.title("Satellite Image Classification with Deep Learning")
main.geometry("1300x1200")

global filename
global deep_learning_acc
global classifier
global X,Y


labels = ['Urban Land','Agricultural Land','Range Land','Forest Land']

def upload():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");

def extractFeatures():
    global X,Y
    text.delete('1.0', END)
    X = np.load('model/X.txt.npy')
    Y = np.load('model/Y.txt.npy')

    text.insert(END,"Total Images Found in dataset : "+str(len(X))+"\n")
    

def runCNN():
    global X,Y
    global neural_network_acc
    global classifier
    Y1 = to_categorical(Y)
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        classifier.load_weights("model/model_weights.h5")
        classifier._make_predict_function()   
        print(classifier.summary())
        f = open('model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        neural_network_acc = acc[19] * 100
        text.insert(END,"CNN Neural Networks Accuracy : "+str(neural_network_acc)+"\n")
    else:
        classifier = Sequential()
        classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(output_dim = 256, activation = 'relu'))
        classifier.add(Dense(output_dim = 4, activation = 'softmax'))
        print(classifier.summary())
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = classifier.fit(X, Y1, batch_size=32, epochs=20, shuffle=True, verbose=2)
        classifier.save_weights('model/model_weights.h5')            
        model_json = classifier.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        f = open('model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        neural_network_acc = acc[19] * 100
        text.insert(END,"CNN Neural Networks Accuracy : "+str(neural_network_acc)+"\n")
    
    
def graph():
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    acc = data['accuracy']
    loss = data['loss']

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy/Loss')
    plt.plot(acc, 'ro-', color = 'green')
    plt.plot(loss, 'ro-', color = 'blue')
    plt.legend(['Accuracy', 'Loss'], loc='upper left')
    plt.title('Deep Learning CNN Accuracy & Loss Graph')
    plt.show()

def predict():
    filename = filedialog.askopenfilename(initialdir="sampleImages")
    image = cv2.imread(filename)
    img = cv2.resize(image, (64,64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,64,64,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = classifier.predict(img)
    predict = np.argmax(preds)

    img = cv2.imread(filename)
    img = cv2.resize(img, (800,400))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([36, 25, 25])
    upper = np.array([70, 255, 255])

    mask = cv2.inRange (hsv, lower, upper)
    contours,temp = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        if len(contours[i]) > 10:
            red_area = contours[i] 
            x, y, w, h = cv2.boundingRect(red_area)
            cv2.rectangle(img,(x, y),(x+w, y+h),(0, 0, 255), 2)
          
    cv2.putText(img, 'Satellite Image Classified as : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
    cv2.imshow('Satellite Image Classified as : '+labels[predict], img)
    cv2.waitKey(0)
    
font = ('times', 14, 'bold')
title = Label(main, text='Detecting Deforestation from Satellite Images')
title.config(bg='yellow3', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Satellite Images Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=460,y=100)

featuresButton = Button(main, text="Extract Features from Images", command=extractFeatures)
featuresButton.place(x=50,y=150)
featuresButton.config(font=font1) 

cnnButton = Button(main, text="Train CNN Algorithm", command=runCNN)
cnnButton.place(x=310,y=150)
cnnButton.config(font=font1) 

graphbutton = Button(main, text="Accuracy Graph", command=graph)
graphbutton.place(x=50,y=200)
graphbutton.config(font=font1) 

predictb = Button(main, text="Upload Test Image & Clasify", command=predict)
predictb.place(x=310,y=200)
predictb.config(font=font1) 


font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='burlywood2')
main.mainloop()
