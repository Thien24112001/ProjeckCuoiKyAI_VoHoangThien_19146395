from tkinter import *
from tkinter.filedialog import Open, SaveAs
import numpy as np
import cv2 as cv
from keras.models import model_from_json
import cv2

model_architecture = "breast_N_C.json"
model_weights = "breast_N_C.h5"
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)
mydict = ['Cancer','Normal']


def LOAD_IMG(file_in):
    IMG_SIZE = 224
    img_array = cv2.cvtColor(file_in,cv2.COLOR_BGR2GRAY)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
# Load haarcascade
class Main(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.initUI()
  
    def initUI(self):
        self.parent.title("NHẬN DIỆN UNG THƯ VÚ")
        self.pack(fill=BOTH, expand=1)
  
        menubar = Menu(self.parent)
        self.parent.config(menu=menubar)
  
        fileMenu = Menu(menubar)
        fileMenu.add_command(label="Open Camera", command=self.onOpenCam)
        fileMenu.add_command(label="Open", command=self.onOpen)
        fileMenu.add_separator()
        fileMenu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="Menu", menu=fileMenu)
        #self.txt = Text(self)
        #self.txt.pack(fill=BOTH, expand=1)
    def onOpen(self):
        global ftypes
        ftypes = [('Images', '*.jpg *.tif *.bmp *.gif *.png')]
        dlg = Open(self, filetypes = ftypes)
        fl = dlg.show()
  
        if fl != '':
            global img
            global imgin
            global new_array
            imgin = cv2.imread(fl)
            img = LOAD_IMG(imgin)
            prediction = np.argmax(model.predict(img),axis=-1)
            result = mydict[prediction[0]]
            cv.namedWindow('Test', cv.WINDOW_AUTOSIZE)
            imgout = cv2.resize(imgin, (350, 350))
            cv2.putText(imgout,result,(15,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
            cv.imshow('Test',imgout)
            if cv.waitKey(1) & 0xFF == ord('q'):
                cv.destroyAllWindows()
    def onOpenCam(self):
        # Enter the path to your test image
        cap = cv.VideoCapture(0)
        while 1:
            ret, frame = cap.read()
            if not ret:
                print('unavailable')
            # cv.imwrite('data.jpg',frame)
            # img = cv.imread('data.jpg')
            frame_array = LOAD_IMG(frame)
            prediction = np.argmax(model.predict(frame_array),axis=-1)
            result = mydict[prediction[0]]
            cv.namedWindow('RealTime Test', cv.WINDOW_AUTOSIZE)
            cv2.putText(frame,result,(15,60),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,255))
            cv.imshow('RealTime Test',frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv.destroyAllWindows()
                break
root = Tk()
Main(root)
root.geometry("480x480+100+100")
root.mainloop()