import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import tkinter as tk
from tkinter import ttk

path = 'images'
image = []
classname = []
classlist = os.listdir(path)
print(classlist)

for cl in classlist:
    curimg = cv2.imread(f'{path}/{cl}')
    image.append(curimg)
    classname.append(os.path.splitext(cl)[0])
print(classname)


def findencodings(image):
    encodelist = []
    for img in image:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist


def markattendance(name):
    with open('attendance.csv', 'r+') as f:
        myDatalist = f.readlines()
        namelist = []
        for line in myDatalist:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name}, {dtstring}')


encodelistknown = findencodings(image)
print(len(encodelistknown))

# Create a list to store recognized attendees
recognized_attendees = []


# Create a function to update the "Display Attendees" section
def update_attendees_display():
    attendees_display.delete(1.0, tk.END)  # Clear the existing display
    for attendee in recognized_attendees:
        attendees_display.insert(tk.END, attendee + "\n")


# Create a function to start the facial recognition process
def start_recognition():
    cap = cv2.VideoCapture(0)

    def recognize_faces():
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)

        facescurframe = face_recognition.face_locations(imgS)
        encodescurframe = face_recognition.face_encodings(imgS, facescurframe)

        for encodeface, faceloc in zip(encodescurframe, facescurframe):
            matches = face_recognition.compare_faces(encodelistknown, encodeface)
            facdis = face_recognition.face_distance(encodelistknown, encodeface)

            matchindex = np.argmin(facdis)

            if matches[matchindex]:
                name = classname[matchindex].upper()
                if name not in recognized_attendees:
                    recognized_attendees.append(name)
                    update_attendees_display()

                y1, x2, y2, x1 = faceloc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markattendance(name)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return
        root.after(10, recognize_faces)  # Schedule recognition after a short delay

    # Start recognizing faces
    recognize_faces()


# Create a tkinter window
root = tk.Tk()
root.title("Facial Recognition")

# Create a label and button in the GUI
label = ttk.Label(root, text="Click the button to start facial recognition:")
label.pack(pady=10)
start_button = ttk.Button(root, text="Start Recognition", command=start_recognition)
start_button.pack()

# Create a "Display Attendees" section
attendees_label = ttk.Label(root, text="Recognized Attendees:")
attendees_label.pack()
attendees_display = tk.Text(root, height=10, width=30)
attendees_display.pack()

# Add a quit button to close the GUI
quit_button = ttk.Button(root, text="Quit", command=root.destroy)
quit_button.pack()

# Run the GUI main loop
root.mainloop()
