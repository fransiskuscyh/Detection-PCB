<h1>"Detection PCB"</h1>

<p>Detection PCB pass or reject using python</p>



## Instalasi
```bash
create database using step "createDatabase.txt"
git clone https://github.com/fransiskuscyh/Detection-PCB.git
cd Detection-PCB
pip install numpy, tensorflow, scikit-learn, matplotlib, customtkinter, tkinter, mysql-connector-python
python modelv5.py #run models
python konvertModelv3.py #convert models to tflite
python AppDetectPcb.py #run for windows
python AppDetectPcbLow.py #run for windows low resolution
python RaspberryDetectPcb.py #run for raspberry pi
