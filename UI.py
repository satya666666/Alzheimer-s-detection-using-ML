from tkinter import *
from tkinter.filedialog import askopenfile 
from PIL import ImageTk,Image

win = Tk()
win.title("Final Year Project")
win.geometry("800x700")

file_selected=StringVar()


def openfile():

    f = askopenfile(mode ='r', filetypes =[('Image Files', '*.jpg')])

    file_selected.set(f.name)


def getresult():
    



    print("YOU ARE NOT SUFFERING AD")






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
                      width=30,height=2)

train_button.place(x=120, y=100)    



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




footer = Label(win,
               text="Saurav Sharma | Vaibhav Gupta | Satyam Gupta | Vikash Kumar Singh",
               fg="white",
               bg="black")

footer.place(x=230,y=650)               

  









win.mainloop()