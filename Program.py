import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import tensorflow as tf
import keras
import glob
import cv2
from tqdm import tqdm

import os
import pickle










####################----UI Design -------------#####################



from tkinter import *
from tkinter.filedialog import askopenfile 
from PIL import ImageTk,Image

win = Tk()
win.title("Final Year Project")
win.geometry("800x700")

file_selected=StringVar()
training_status=StringVar()
patient_report=StringVar()



################# -------- Building Model ----------#


def trainmodel():
    global model_cnn
    global id_to_label_dict
    

    images = []
    labels = [] 
    
    
    training_status.set("Processing Dataset this will take a while...")
    
      
    
    
    for dir_path in glob.glob("Alzheimer_s Dataset/train/MildDemented/"):
        label = "MildDemented"
        for image_path in tqdm(glob.glob(os.path.join(dir_path, "*.jpg"))):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (45, 45))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            images.append(image)
            labels.append(label)
    for dir_path in glob.glob("Alzheimer_s Dataset/test/MildDemented/"):
        label = "MildDemented"
        for image_path in tqdm(glob.glob(os.path.join(dir_path, "*.jpg"))):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (45, 45))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            images.append(image)
            labels.append(label)
    
    for dir_path in glob.glob("Alzheimer_s Dataset/train/VeryMildDemented/"):
        label = "VeryMildDemented"
        for image_path in tqdm(glob.glob(os.path.join(dir_path, "*.jpg"))):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (45, 45))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            images.append(image)
            labels.append(label)
    for dir_path in glob.glob("Alzheimer_s Dataset/test/VeryMildDemented/"):
        label = "VeryMildDemented"
        for image_path in tqdm(glob.glob(os.path.join(dir_path, "*.jpg"))):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (45, 45))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            images.append(image)
            labels.append(label)
    
    
    
    
    
    for dir_path in glob.glob("Alzheimer_s Dataset/train/ModerateDemented/"):
        label = "ModerateDemented"
        for image_path in tqdm(glob.glob(os.path.join(dir_path, "*.jpg"))):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (45, 45))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            images.append(image)
            labels.append(label)
    for dir_path in glob.glob("Alzheimer_s Dataset/test/ModerateDemented/"):
        label = "ModerateDemented"
        for image_path in tqdm(glob.glob(os.path.join(dir_path, "*.jpg"))):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (45, 45))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            images.append(image)
            labels.append(label)
    
    
    
    for dir_path in glob.glob("Alzheimer_s Dataset/train/NonDemented/"):
        label = "NonDemented"
        for image_path in tqdm(glob.glob(os.path.join(dir_path, "*.jpg"))):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (45, 45))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            images.append(image)
            labels.append(label)
    for dir_path in glob.glob("Alzheimer_s Dataset/test/NonDemented/"):
        label = "NonDemented"
        for image_path in tqdm(glob.glob(os.path.join(dir_path, "*.jpg"))):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (45, 45))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            images.append(image)
            labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    
    

    
    
    
    label_to_id_dict = {v:i for i,v in enumerate(np.unique(labels))}
    id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
    
    print(id_to_label_dict)
    
    label_ids = np.array([label_to_id_dict[x] for x in labels])
    
    print(label_ids)
    
    #images.shape, label_ids.shape, labels.shape
    
    

    
    
        
    
    from sklearn.model_selection import train_test_split  
    X_train, X_test, y_train, y_test = train_test_split(images,label_ids, test_size = 0.20)
    
    #X_train
    
    #y_train
    
    #Normalize color values to between 0 and 1
    X_train = X_train/255
    X_test = X_test/255
    
    
    #Make a flattened version for some of our models
    X_flat_train = X_train.reshape(X_train.shape[0], 45*45*3)
    X_flat_test = X_test.reshape(X_test.shape[0], 45*45*3)
    
    #y_train
    
    
    #One Hot Encode the Output
    y_train = keras.utils.to_categorical(y_train, 4)
    y_test = keras.utils.to_categorical(y_test, 4)
    
    print('Original Sizes:', X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print('Flattened:', X_flat_train.shape, X_flat_test.shape,type(y_train[0][0]), y_test[0])
    
    
    
    print(X_train[0].shape)
    plt.imshow(X_train[0])
    plt.show()
    
    
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

    
    
    model_cnn = Sequential()
    # First convolutional layer, note the specification of shape
    model_cnn.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(45, 45, 3)))
    model_cnn.add(Conv2D(64, (3, 3), activation='relu'))
    model_cnn.add(MaxPooling2D(pool_size=(2, 2)))
    model_cnn.add(Dropout(0.25))
    model_cnn.add(Flatten())
    model_cnn.add(Dense(128, activation='relu'))
    model_cnn.add(Dropout(0.5))
    model_cnn.add(Dense(4, activation='softmax'))
    
    model_cnn.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    
    print(model_cnn.summary())
    
    
    
    model_cnn.fit(X_train, y_train,
              batch_size=128,
              epochs=30,
              verbose=1,
              validation_data=(X_test, y_test))
    score = model_cnn.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    with open("model_pickle.pkl","wb") as f:
        pickle.dump(model_cnn,f)
    with open("i_to_l.pkl","wb") as f:
        pickle.dump(id_to_label_dict,f)    
    
    
    training_status.set("Training Complete...")


    



def openfile():

    f = askopenfile(mode ='r', filetypes =[('Image Files', '*.jpg')])

    file_selected.set(f.name)
    
    
    
    
def savedmodel():
    global model_cnn
    global id_to_label_dict
    
    if "model_pickle.pkl" in os.listdir("."):
        with open("model_pickle.pkl","rb") as f:
            model_cnn=pickle.load(f)
            
        with open("i_to_l.pkl","rb") as f:
            id_to_label_dict=pickle.load(f)    
        training_status.set("Training Complete(With Saved Model)...")    
    
    else:
        trainmodel()


def getresult():
    global model_cnn
    global id_to_label_dict
    global patient_report
    
    

    

    image = cv2.imread(file_selected.get(),cv2.IMREAD_COLOR)
    
    image = cv2.resize(image, (45, 45))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    test=image.reshape(1,45,45,3)
    test=np.array(test)
    
    for i in model_cnn.predict(test)[0]:
        print(i)
    temp=model_cnn.predict(test)[0]
    temp=list(temp)
    temp=temp.index(max(temp))
    print(id_to_label_dict[temp])
    patient_report.set(id_to_label_dict[temp])
        
        






################ -------------Header-----------------------#############


header = Label(win,
               text="UNAUTHORIZED USE NOT PERMITTED!!!" ,
               fg="red")

header.pack()    


############----------------Training Section ----------------------############


train_img = ImageTk.PhotoImage(Image.open("icons/train.png"))

train_icon = Label(win,image=train_img,height=50)

train_icon.place(x=12,y=100)

train_button = Button(win,
                      text="Train Model", 
                      width=30,
                      height=2,
                      command=savedmodel)

train_button.place(x=120, y=100)    

train_text = Label(win,
                    textvariable=training_status)

train_text.place(x=350,y=100) 



####################-----------------------Image Selector -------------###############

select_img = ImageTk.PhotoImage(Image.open("icons/selectfile.png"))

select_icon = Label(win, image=select_img,height=50)

select_icon.place(x=22,y=200)

select_button = Button(win,
                       text="Select the Image file to test",
                       width=30,
                       height=2, 
                       command=openfile)
                       

select_button.place(x=120,y=200)                         

select_text = Label(win,
                    textvariable=file_selected)

select_text.place(x=350,y=209)    




#####################----------------Result Section--------------------------####################


result_img = ImageTk.PhotoImage(Image.open("icons/result.png"))

result_icon = Label(win, image=result_img,height=50)

result_icon.place(x=22,y=300)

result_button = Button(win, 
                       text="Get Results",
                       width=30,
                       height=2,
                       command=getresult)

result_button.place(x=120, y=300)

result_text = Label(win,
                    textvariable=patient_report)

result_text.place(x=350,y=300) 




footer = Label(win,
               text="Saurav Sharma | Vaibhav Gupta | Satyam Gupta | Vikash Kumar Singh",
               fg="white",
               bg="black")

footer.place(x=230,y=650)               

  









win.mainloop()



#######################----------- End Here ------------################################






