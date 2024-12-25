import subprocess
import tkinter as tk
from tkinter import Label
import numpy as np
from PIL import Image, ImageTk
import tensorflow.lite as tflite
import os
import mysql.connector
import time
import cv2

db_connection = mysql.connector.connect(
    host="detectpcb.crquwg0oa33g.us-east-1.rds.amazonaws.com",  
    user="admin",       
    password="Useradminpassword",       
    database="raspberry" 
)

db_cursor = db_connection.cursor()

interpreter = tflite.Interpreter(model_path="pcb_detection_modelv5.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def save_to_database(status, image_data):
    if status == "Pass":
        query = """
        INSERT INTO bypass_bypassmode (name, description, image, status) 
        VALUES (%s, %s, %s, %s)
        """
        description = "PCB memenuhi standar"
    else: 
        query = """
        INSERT INTO byreject_byrejectmode (name, description, image, status) 
        VALUES (%s, %s, %s, %s)
        """
        description = "PCB tidak memenuhi standar"

    db_cursor.execute(query, ("PCB", description, image_data, status))
    db_connection.commit()

def predict_image(image_path):
    image = Image.open(image_path).resize(
        (input_details[0]['shape'][1], input_details[0]['shape'][2])
    )
    input_data = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    status = "Pass" if np.argmax(output_data) == 0 else "Reject"
    
    _, buffer = cv2.imencode('.jpg', np.array(image))
    image_data = buffer.tobytes()
    save_to_database(status, image_data)
    
    return status

def capture_and_predict():
    image_path = "temp_pcb_image.jpg"
    
    subprocess.run(['libcamera-still', '-o', image_path, '--immediate'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if os.path.exists(image_path):
        status = predict_image(image_path)
        status_label.config(text=f"Status: {status}")
        update_display(image_path)
        os.remove(image_path)
    
    root.after(2000, capture_and_predict)

def update_display(image_path):
    image = Image.open(image_path).resize((300, 200))
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.photo = photo

def on_close():
    db_cursor.close()
    db_connection.close()
    root.destroy()

root = tk.Tk()
root.title("PCB Pass/Reject Detection")
root.geometry("400x400")

status_label = Label(root, text="Status: N/A", font=("Arial", 14))
status_label.pack(pady=10)

image_label = Label(root)
image_label.pack(pady=10)

root.after(1000, capture_and_predict)
root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
