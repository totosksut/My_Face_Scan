import PIL
from PIL import Image,ImageTk
import pytesseract
from tkinter import *
from threading import Timer,Thread,Event
import cv2, sys, numpy, os
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'
width, height = 800, 600

#stamp found
st_name = []

# Create a list of images and a list of corresponding names
(images, labels, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1
(width_face, height_face) = (130, 100)

(images, labels) = [numpy.array(lis) for lis in [images, labels]]
model = cv2.face.LBPHFaceRecognizer_create()

model.train(images, labels)
face_cascade = cv2.CascadeClassifier(haar_file)


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

solve = None
root = Tk()
root.title("Face Scan")
root.bind('<Escape>', lambda e: root.quit())
lmain = Label(root)
lmain.pack()


def show_frame():
    global solve
    
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = PIL.Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    
    solve = lmain.after(10, show_frame)
    

def callback():
    print("clicked2") 
    lmain.after_cancel(solve)
    print("list is ",set(st_name))

    top1 = lmain.top = Toplevel(root)
    text_box = Text(top1, width=20, height=20, font=("Helvetica", 12))
    text_box.pack()
    scrollbar = Scrollbar(top1, orient="vertical")
    scrollbar.config(command=text_box.yview)
    text_box.config(yscrollcommand=scrollbar.set)

    seq = 1
    for x in set(st_name):
        text_box.insert(END, str(seq)+' '+str(x)+'\n')
        seq+=1

def openWebcam():
    lmain.after_cancel(solve)
    show_frame()

def face_rec():
    print("face rec")
    lmain.after_cancel(solve)
    global datasets
    # Create a list of images and a list of corresponding names
    
    global images
    global labels
    global names
    
    (images, labels, names, id) = ([], [], {}, 0)
    for (subdirs, dirs, files) in os.walk(datasets):
        for subdir in dirs:
            names[id] = subdir
            subjectpath = os.path.join(datasets, subdir)
            for filename in os.listdir(subjectpath):
                path = subjectpath + '/' + filename
                label = id
                images.append(cv2.imread(path, 0))
                labels.append(int(label))
            id += 1
    (width_face, height_face) = (130, 100)

    (images, labels) = [numpy.array(lis) for lis in [images, labels]]
    global model
    model = None
    global face_cascade
    face_cascade = None
    model = cv2.face.LBPHFaceRecognizer_create()

    model.train(images, labels)
    face_cascade = cv2.CascadeClassifier(haar_file)

    face_rec_start()
    

def create_db():
    lmain.after_cancel(solve)
    top = lmain.top = Toplevel(root)
    Label(top, text="Enter ID Student" , width=30).pack()

    lmain.e = Entry(top)
    lmain.e.pack(padx=15)

    b = Button(top, text="OK", command=create_db_face)
    b.pack(pady=5)

def face_rec_start():
    global solve
    global st_name
    global images
    global labels
    global names
    
    (_, frame) = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width_face, height_face))
        #Try to recognize the face
        prediction = model.predict(face_resize)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if prediction[1]<100:
                cv2.putText(frame,'%s - %.0f' % (names[prediction[0]],prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
                st_name.append(names[prediction[0]])
        else:
                cv2.putText(frame,'Unknown',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))

    #cv2.imshow('OpenCV', frame)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = PIL.Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    
    solve = lmain.after(10, face_rec_start)

def create_db_face():
    print("save ",lmain.e.get())
    sub_data = lmain.e.get()
    if sub_data : 
        lmain.top.destroy()
        path = os.path.join(datasets, str(sub_data))
        if not os.path.isdir(path):
            print('NOT FOUND')
            os.mkdir(path)
        (width, height) = (130, 100)    # defining the size of images 
        print('PASS STEP1')

        count = 1
        while count < 31: 
            (_, im) = cap.read()
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 4)
            for (x,y,w,h) in faces:
                cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
                face = gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (width, height))
                cv2.imwrite('%s/%s.png' % (path,count), face_resize)
            count += 1
            
            cv2.imshow('Save Face', im)
            key = cv2.waitKey(150)
            if key == 27:
                break
        cv2.destroyAllWindows();

#button = Button(root, text='create', width=25, command=root.destroy)
button = Button(root, text='play', width=25, command=openWebcam, bg = "yellow")
button.pack(side="left")
btnRec = Button(root, text='Stop and Result', width=25, command=callback, bg = "red")
btnRec.pack(side="left")
btnRecFace = Button(root, text='Face Rec', width=25, command=face_rec, bg = "darkorange")
btnRecFace.pack(side="left")
btnCreate = Button(root, text='Create Face', width=25, command=create_db, bg = "green" ,fg='white')
btnCreate.pack(side="right")

show_frame()
root.mainloop()
