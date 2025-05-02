from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter import ttk
#importing require python packages
import os
import numpy as np
import pandas as pd
from sklearn import linear_model
from scipy import stats
import matplotlib.pyplot as plt #use to visualize dataset vallues

import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout, LSTM
from keras.callbacks import ModelCheckpoint
import os
import pickle
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Convolution2D
from sklearn.metrics import mean_squared_error
from math import sqrt


main = Tk()
main.title("Remaining Shelf-Life Estimation of Fresh Fruits and Vegetables During Transportation")
main.geometry("1300x1200")

global X, Y, dataset, propose_error, extension_error, scaler, scaler1

def uploadDataset():
    global dataset
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n");
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset.head()))

def preprocessDataset():
    global dataset, X, Y
    text.delete('1.0', END)
    dataset['Diff'] = dataset['Temp'].diff()
    dataset.fillna(0, inplace = True)
    data = dataset.values
    X = data[:,0:2]
    Y = data[:,data.shape[1]-1]
    text.insert(END,"Dataset Processing RSL (Remaining Shelf Life) Calculation Completed\n\n")
    text.insert(END,"RSL = "+str(dataset['Diff']))

def SLEMANOVA():
    global X, Y, propose_error, dataset
    text.delete('1.0', END)
    X = sm.add_constant(X) # adding a constant
    olsmod = sm.OLS(Y, X).fit()
    predict = olsmod.predict(X)
    propose_error = sqrt(mean_squared_error(Y, predict.ravel()))
    lm = ols(formula = 'Diff ~ C(Temp)',data=dataset).fit()
    table = sm.stats.anova_lm(lm, typ=2)

    print("ANOVA Summary\n"+str(olsmod.summary())+"\n\n")
    print(str(table)+"\n\n")
    text.insert(END,"Propose SLEM Model Error : "+str(propose_error)+"\n\n")
    for i in range(0, 10):
        original = Y[i]*30
        predicted = predict[i]*30
        if original > 50:
            original = original / 10
        if predicted > 50:
            predicted = predicted / 10
        text.insert(END,"Observed Life : "+str(int(abs(original)))+" Days Predicted Life : "+str(int(abs(predicted)))+" Days\n")
    plt.figure(figsize=(5,3))
    plt.plot(Y, color = 'red', label = 'Observed Life')
    plt.plot(predict, color = 'green', label = 'Predicted Life')
    plt.title('Propose SLEM ANNOVA Prediction Graph')
    plt.xlabel('Test Data')
    plt.ylabel('Life Values')
    plt.legend()
    plt.show()        

#function to calculate accuracy and prediction sales graph
def calculateMetrics(algorithm, predict, test_labels):
    global extension_error
    extension_error = sqrt(mean_squared_error(test_labels, predict))
    predict = predict.reshape(-1, 1)
    predict = scaler1.inverse_transform(predict)
    test_label = scaler1.inverse_transform(test_labels)
    predict = predict.ravel()
    test_label = test_label.ravel()    
    text.insert(END,"Extension CNN Model Error : "+str(extension_error)+"\n\n")
    for i in range(0, 10):
        original = test_label[i]*30
        predicted = predict[i]*30
        if original > 50:
            original = original / 10
        if predicted > 50:
            predicted = predicted / 10
        text.insert(END,"Observed Life : "+str(int(abs(original)))+" Days Predicted Life : "+str(int(abs(predicted)))+" Days\n")
    plt.figure(figsize=(5,3))
    plt.plot(test_label, color = 'red', label = 'Observed Life')
    plt.plot(predict, color = 'green', label = 'Predicted Life')
    plt.title(algorithm+' SLEM Prediction Graph')
    plt.xlabel('Test Data')
    plt.ylabel('Life Values')
    plt.legend()
    plt.show()    

def CNNSLEM():
    global X, Y, extension_error, scaler, scaler1
    text.delete('1.0', END)
    scaler = MinMaxScaler(feature_range = (0, 1))
    scaler1 = MinMaxScaler(feature_range = (0, 1))
    X = scaler.fit_transform(X)
    Y = Y.reshape(-1, 1)
    Y = scaler1.fit_transform(Y)

    #now train & plot CNN crop yield prediction
    X = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
    #training CNN model
    cnn_model = Sequential()
    cnn_model.add(Convolution2D(32, (1 , 1), input_shape = (X.shape[1], X.shape[2], X.shape[3]), activation = 'relu'))
    cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
    cnn_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
    cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(units = 256, activation = 'relu'))
    cnn_model.add(Dense(units = 1))
    cnn_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    if os.path.exists("model/cnn_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
        cnn_model.fit(X, Y, batch_size = 8, epochs = 1000, validation_data=(X, Y), callbacks=[model_check_point], verbose=1)
    else:
        cnn_model.load_weights("model/cnn_weights.hdf5")
    predict = cnn_model.predict(X)
    calculateMetrics("Extension CNN", predict, Y)#call function to plot LSTM crop yield prediction

def graph():
    labels = ['Propose SLEM Model Error', 'Extension CNN Error']
    height = (propose_error, extension_error)
    bars = labels
    y_pos = np.arange(len(bars))
    plt.figure(figsize = (4, 3)) 
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Algorithm Names")
    plt.ylabel("Error %")
    plt.title("Propose & Extension Error Comparison Graph")
    plt.xticks()
    plt.tight_layout()
    plt.show()

def close():
    global main
    main.destroy()
    

font = ('times', 15, 'bold')
title = Label(main, text='Remaining Shelf-Life Estimation of Fresh Fruits and Vegetables During Transportation')
title.config(bg='bisque', fg='purple1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')

uploadButton = Button(main, text="Upload Strawberry Shelflife Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)

processButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
processButton.place(x=350,y=100)
processButton.config(font=font1)

proposeButton = Button(main, text="Propose Shelf Life Estimation + ANOVA", command=SLEMANOVA)
proposeButton.place(x=650,y=100)
proposeButton.config(font=font1)

extensionButton = Button(main, text="Extension CNN SLEM", command=CNNSLEM)
extensionButton.place(x=50,y=150)
extensionButton.config(font=font1)

graphButton = Button(main, text="Model Error Comparison Graph", command=graph)
graphButton.place(x=350,y=150)
graphButton.config(font=font1)

closeButton = Button(main, text="Exit", command=close)
closeButton.place(x=650,y=150)
closeButton.config(font=font1)

font1 = ('times', 13, 'bold')
text=Text(main,height=20,width=120)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)

main.config(bg='cornflower blue')
main.mainloop()
