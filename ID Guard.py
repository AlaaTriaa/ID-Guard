import tkinter as tk
from tkinter import *
import cv2
import os
from PIL import Image, ImageTk
import numpy as np
import mysql.connector
from tkinter import messagebox
import winsound
import time




window = tk.Tk()
window.title("ID-GUARD")
window.resizable(0, 0)
window.config(bg="#021636")

load1 = Image.open("header.png")
photo1 = ImageTk.PhotoImage(load1)


def openNewWindow():
    # Toplevel object which will
    # be treated as a new window
    newWindow = Toplevel(window)

    # sets the title of the
    # Toplevel widget
    newWindow.title("New Window")

    # sets the geometry of toplevel
    newWindow.geometry("200x200")

    # A Label widget to show in toplevel
    Label(newWindow,
          text="description..................................................................").pack()

header = tk.Button(window, image=photo1,command=openNewWindow)
header.place(x=180, y=10)







def clock():
    txt=time.strftime("%d %b \n %H:%M:%S")
    time_lbl.config(text = txt)
    time_lbl.after(200, clock)

time_lbl = Label(window, font = ('DS-Digital', 20, 'bold'),foreground = 'white',bg="#021636")
time_lbl.place(x=10,y=5)

clock()

canvas1 = Canvas(window, width=500, height=260,bg="#021636",highlightbackground="#021636")
canvas1.place(x=5, y=180)

l1 = tk.Button(canvas1, text="Name", font=("Algerian", 15),bg="#021636",fg="white")
l1.place(x=5, y=5)
t1 = tk.Entry(canvas1, width=50, bd=5)
t1.place(x=150, y=10)

l2 = tk.Button(canvas1, text="Age", font=("Algerian", 15),bg="#021636",fg="white")
l2.place(x=5, y=50)
t2 = tk.Entry(canvas1, width=50, bd=5)
t2.place(x=150, y=55)

l3 = tk.Button(canvas1, text="Address", font=("Algerian", 15),bg="#021636",fg="white")
l3.place(x=5, y=100)
t3 = tk.Entry(canvas1, width=50, bd=5)
t3.place(x=150, y=105)


def time():
    string = strftime('%H:%M:%S %p')
    lbl.config(text = string)
    lbl.after(1000, time)

def train_classifier():
    data_dir = "C:/Users/Alaa Triaa/PycharmProjects/Face Recognition System/data"
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])

        faces.append(imageNp)
        ids.append(id)

    ids = np.array(ids)

    # Train and save classifier
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")
    messagebox.showinfo("Result", "Model trained successful")


b1 = tk.Button(canvas1, text="Training", font=("Algerian", 20), bg='#730314', fg='white', command=train_classifier)
b1.place(x=10, y=155)


def generate_dataset():
    if (t1.get() == "" or t2.get() == "" or t3.get() == ""):
        messagebox.showinfo("Result", "Please provide complete details of the user")
    else:
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="admin",
            database="id-guard"
        )
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * from data")
        myresult = mycursor.fetchall()

        id = 1
        for x in myresult:
            id += 1

        sql = "insert into data(id,name,age,address) values(%s, %s, %s, %s)"
        val = (id, t1.get(), t2.get(), t3.get())
        mycursor.execute(sql, val)
        mydb.commit()

        face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        def face_cropped(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)
            # scaling factor = 1.3
            # minimum neighbor = 5

            if faces is ():
                return None
            for (x, y, w, h) in faces:
                cropped_face = img[y:y + h, x:x + w]
            return cropped_face

        cap = cv2.VideoCapture(0)
        img_id = 0

        while True:
            ret, frame = cap.read()
            if face_cropped(frame) is not None:
                img_id += 1
                face = cv2.resize(face_cropped(frame), (200, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                file_name_path = "data/user." + str(id) + "." + str(img_id) + ".jpg"
                cv2.imwrite(file_name_path, face)
                cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("Cropped face", face)

            if cv2.waitKey(1) == 13 or int(img_id) == 10:  # 13 is the ASCII character of Enter
                break

        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Result", "Generating dataset is completed. ")


b2 = tk.Button(canvas1, text="Generate dataset", font=("Algerian", 20), bg='#730314', fg='white', command=generate_dataset)
b2.place(x=175, y=155)

load2 = Image.open("canvas2.png")
photo2 = ImageTk.PhotoImage(load2)

canvas2 = Canvas(window, width=500, height=250,bg="#021636",highlightbackground="#021636")
canvas2.place(x=5, y=400)
#canvas2.create_image(250, 125, image=photo2)
def capture_stranger():
    canvas3 = Canvas(window, width=280, height=420, bg="#021636")
    canvas3.place(x=495, y=205)

    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False

    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    # scaling factor = 1.3
    # minimum neighbor = 5

    if faces is ():
        print("No face detected..")
        return None
    #else:
        #for (x, y, w, h) in faces:
            #cropped_face = frame[y:y + h, x:x + w]
        #cv2.imwrite("captured_image.jpg", cropped_face)

        #load = Image.open("captured_image.jpg")
        #photo = ImageTk.PhotoImage(load)

        #img = Label(canvas3, image=photo, width=200, height=200)
        #img.image = photo

        #img.place(x=0, y=5)

        #cap.release()
    a = tk.Label(canvas3, text="Stranger", font=("Algerian", 20))
    a.place(x=5, y=250)

def capture_image():
    canvas3 = Canvas(window, width=280, height=420, bg="#021636")
    canvas3.place(x=495, y=205)


    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False

    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3,5)
    # scaling factor = 1.3
    # minimum neighbor = 5

    if faces is ():
        print("No face detected..")
        return None
    else:
        for (x, y, w, h) in faces:
            cropped_face = frame[y:y + h, x:x + w]
        cv2.imwrite("captured_image.jpg", cropped_face)

        load = Image.open("captured_image.jpg")
        photo = ImageTk.PhotoImage(load)

        # Labels can be text or images
        img = Label(canvas3, image=photo, width=200, height=200)
        img.image = photo

        img.place(x=45, y=5)

        cap.release()

    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    for (x, y, w, h) in faces:
        id, pred = clf.predict(gray[y:y + h, x:x + w])
        # confidence = int(100*(1-pred/300))

        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="admin",
            database="id-guard"
        )
        mycursor = mydb.cursor()

        mycursor.execute("select name,age,address from data where id=" + str(id))
        s = mycursor.fetchall()

        a1 = tk.Label(canvas3, text="Name = ", font=("Algerian", 10),bg="#021636",fg="white")
        a1.place(x=5, y=250)

        b1 = tk.Label(canvas3, text=s[0][0], font=("Algerian", 10),bg="#021636",fg="white")
        b1.place(x=100, y=250)

        c1 = tk.Label(canvas3, text="Age = ", font=("Algerian", 10),bg="#021636",fg="white")
        c1.place(x=5, y=300)

        d1 = tk.Label(canvas3, text=s[0][1], font=("Algerian", 10),bg="#021636",fg="white")
        d1.place(x=100, y=300)

        e1 = tk.Label(canvas3, text="Address = ", font=("Algerian", 10) ,bg="#021636",fg="white")
        e1.place(x=5, y=350)

        f1 = tk.Label(canvas3, text=s[0][2], font=("Algerian", 10),bg="#021636",fg="white")
        f1.place(x=150, y=350)


b3 = tk.Button(canvas2, text="  User's Info  ", font=("Algerian", 20), bg="#013b29", fg="white", command=capture_image)
b3.place(x=5, y=50)


def detect_face():
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)

        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            id, pred = clf.predict(gray_img[y:y + h, x:x + w])
            confidence = int(100 * (1 - pred / 300))

            mydb = mysql.connector.connect(
                host="localhost",
                user="root",
                passwd="admin",
                database="id-guard"
            )
            mycursor = mydb.cursor()

            mycursor.execute("select name from data where id=" + str(id))
            s = mycursor.fetchone()  # tuple

            s = '' + ''.join(s)  # string

            if confidence > 73:
                cv2.putText(img,s, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            else:
                cv2.putText(img, "UNKNOWN", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                print("Alert , there is a stranger ")
                winsound.PlaySound('alert.wav', winsound.SND_ASYNC)
                capture_stranger();


        return img

    # loading classifier
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, img = video_capture.read(0)
        img = draw_boundary(img, faceCascade, 1.3, 6, (255, 255, 255), "Face", clf)
        cv2.imshow("face Detection", img)

        if cv2.waitKey(1) == 13:
            break
    video_capture.release()
    cv2.destroyAllWindows()


b4 = tk.Button(canvas2, text="Verify Identity", font=("Algerian", 20), bg="#013b29", fg="white",
               command=detect_face)
b4.place(x=5, y=150)



canvas3 = Canvas(window, width=280, height=420,bg="#021636")
canvas3.place(x=495, y=205)

window.geometry("800x680")
window.mainloop()