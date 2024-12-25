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
        self.geometry("640x480")  # Lebih kecil untuk Raspberry Pi
        self.resizable(0, 0)

        self.model = self.load_model()
        self.db_connection, self.db_cursor = self.init_database()

        self.setup_GUI()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)  # Resolusi rendah
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
        self.frame_count = 0  # Counter untuk memproses setiap 10 frame
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
        self.main_frame = ct.CTkFrame(self, width=640, height=480, fg_color="black")
        self.main_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.video_label = ct.CTkLabel(self.main_frame, text="")
        self.video_label.place(relx=0.5, rely=0.4, anchor=tk.CENTER)

        self.result_label = ct.CTkLabel(self.main_frame, font=("Arial", 18), text_color="white")
        self.result_label.place(relx=0.5, rely=0.7, anchor=tk.CENTER)

        self.save_button = ct.CTkButton(self.main_frame, text="Save Result", command=self.save_current_result)
        self.save_button.place(relx=0.5, rely=0.85, anchor=tk.CENTER)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            if self.frame_count % 10 == 0:  # Proses setiap 10 frame
                frame_resized = cv2.resize(frame, (150, 150))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

                self.current_result, annotated_frame = self.predict_pcb_status(frame_rgb)
                self.current_frame = frame_rgb.copy()

                img_pil = Image.fromarray(annotated_frame)
                img_tk = ImageTk.PhotoImage(img_pil)
                self.video_label.configure(image=img_tk)
                self.video_label.image = img_tk

                self.result_label.configure(text=f"Status PCB: {self.current_result}")
                self.result_label.update_idletasks()

        self.video_label.after(100, self.update_frame)  # Interval diperpanjang

    def predict_pcb_status(self, frame_rgb):
        img_array = np.expand_dims(frame_rgb / 255.0, axis=0).astype(np.float32)

        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()

        self.model.set_tensor(input_details[0]['index'], img_array)
        self.model.invoke()
        prediction = self.model.get_tensor(output_details[0]['index'])

        threshold = 0.50
        status = "Reject" if prediction[0][0] > threshold else "Pass"

        annotated_frame = frame_rgb.copy()
        if status == "Reject":
            cv2.rectangle(annotated_frame, (5, 5), (145, 145), (255, 0, 0), 2)

        return status, annotated_frame

    def save_to_database(self, status, frame):
        try:
            timestamp = int(time.time())
            filename = f"{status}_{timestamp}.jpg"
            folder = "images"
            os.makedirs(folder, exist_ok=True)  # Buat folder jika belum ada
            filepath = os.path.join(folder, filename)
            cv2.imwrite(filepath, frame)

            table_name = "bypass_bypassmode" if status == "Pass" else "byreject_byrejectmode"
            description = "PCB memenuhi standar" if status == "Pass" else "PCB tidak memenuhi standar"

            query = f"""
            INSERT INTO {table_name} (`name`, `description`, `image`, `status`)
            VALUES (%s, %s, %s, %s)
            """
            self.db_cursor.execute(query, ("PCB", description, filepath, status))
            self.db_connection.commit()

            messagebox.showinfo("Success", f"Status '{status}' berhasil disimpan ke tabel {table_name}.")
        except mysql.connector.Error as err:
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
