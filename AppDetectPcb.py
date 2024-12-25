import tkinter as tk
import customtkinter as ct
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import cv2
import mysql.connector
from tkinter import messagebox
import time
import os

class DetectPcbApp(ct.CTk):
    def __init__(self):
        super().__init__()
        self.title("Detection PCB")
        self.geometry("1280x660")
        self.resizable(0, 0)

        self.model = self.load_model()
        self.db_connection, self.db_cursor = self.init_database()

        self.setup_GUI()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.update_frame()

    def load_model(self):
        interpreter = tf.lite.Interpreter(model_path="pcb_detection_modelv5.tflite")
        interpreter.allocate_tensors()
        return interpreter

    def init_database(self):
        db_connection = mysql.connector.connect(
            host="localhost", # local
            # host="detectpcb.crquwg0oa33g.us-east-1.rds.amazonaws.com", # aws
            user="root", # local
            # user="admin", # aws
            password="", # local
            # password="Useradminpassword", # aws
            database="raspberry"
        )
        db_cursor = db_connection.cursor()
        return db_connection, db_cursor

    def setup_GUI(self):
        for widget in self.winfo_children():
            widget.destroy()

        self.main_frame = ct.CTkFrame(self, width=1280, height=660, fg_color="black")
        self.main_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.content_frame = ct.CTkFrame(self.main_frame, width=1280, height=600, fg_color="black")
        self.content_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.video_label = ct.CTkLabel(self.content_frame, text="")
        self.video_label.place(relx=0.5, rely=0.4, anchor=tk.CENTER)

        self.result_label = ct.CTkLabel(self.content_frame, font=("Arial", 24), text_color="white")
        self.result_label.place(relx=0.5, rely=0.8, anchor=tk.CENTER)

        self.save_button = ct.CTkButton(self.content_frame, text="Save Result", command=self.save_current_result)
        self.save_button.place(relx=0.5, rely=0.9, anchor=tk.CENTER)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame_resized = cv2.resize(frame, (320, 240))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            self.current_result, annotated_frame = self.predict_pcb_status(frame_rgb)

            self.current_frame = frame_rgb.copy()

            img_pil = Image.fromarray(annotated_frame)
            img_tk = ImageTk.PhotoImage(img_pil)
            self.video_label.configure(image=img_tk)
            self.video_label.image = img_tk

            self.result_label.configure(text=f"Status PCB: {self.current_result}")
            self.result_label.update_idletasks()

        self.video_label.after(30, self.update_frame)

    def predict_pcb_status(self, frame_rgb):
        img_array = cv2.resize(frame_rgb, (150, 150))
        img_array = np.expand_dims(img_array / 255.0, axis=0).astype(np.float32)

        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()

        self.model.set_tensor(input_details[0]['index'], img_array)
        self.model.invoke()
        prediction = self.model.get_tensor(output_details[0]['index'])

        print("Prediction value:", prediction)

        threshold = 0.50
        status = "Reject" if prediction[0][0] > threshold else "Pass"

        annotated_frame = frame_rgb.copy()
        if status == "Reject":
            height, width, _ = annotated_frame.shape
            cv2.rectangle(annotated_frame, (10, 10), (width - 10, height - 10), (255, 0, 0), 2)

        return status, annotated_frame


    def save_to_database(self, status, frame):
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            image_data = buffer.tobytes()  

            if status == "Pass":
                table_name = "bypass_bypassmode"
                description = "PCB memenuhi standar"
            else: 
                table_name = "byreject_byrejectmode"
                description = "PCB tidak memenuhi standar"

            query = f"""
            INSERT INTO {table_name} (`name`, `description`, `image`, `status`)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE 
                status = VALUES(status), 
                description = VALUES(description), 
                image = VALUES(image)
            """
            self.db_cursor.execute(query, ("PCB", description, image_data, status))
            self.db_connection.commit()

            messagebox.showinfo("Success", f"Status '{status}' berhasil disimpan ke tabel {table_name}.")
        except mysql.connector.Error as err:
            print(f"Error: {err}")
            messagebox.showerror("Error", f"Gagal menyimpan status '{status}' ke database: {err}")


    def save_current_result(self):
        if hasattr(self, 'current_result') and hasattr(self, 'current_frame'):
            self.save_to_database(self.current_result, self.current_frame)
        else:
            messagebox.showwarning("Warning", "Tidak ada hasil deteksi yang dapat disimpan.")

    def on_closing(self):
        self.cap.release()
        self.db_cursor.close()
        self.db_connection.close()
        self.destroy()

if __name__ == "__main__":
    app = DetectPcbApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
