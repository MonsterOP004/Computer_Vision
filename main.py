from ultralytics import YOLO
import os
import shutil

model = YOLO("yolo11n.pt")  

results = model("test.png", save=True) 

output_folder = "predictions"
os.makedirs(output_folder, exist_ok=True)  

default_save_dir = model.predictor.save_dir
shutil.move(default_save_dir, output_folder)

print(f"Predicted images are saved in the folder: {output_folder}")