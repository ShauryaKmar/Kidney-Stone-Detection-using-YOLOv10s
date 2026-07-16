from ultralytics import YOLO
import torch

def main():

    model = YOLO("yolov10s.pt")

    model.train(
        data=r"C:\Shaurya\Minor_project\dataset\data.yaml", 
        epochs=100,
        patience=15,             
        imgsz=640,
        batch=16,                 
        device=0,
        workers=4,               
        amp=True,                
        cache=False,            
        project="AI_Demo",
        name="kidney_stone_test_run"
    )

if __name__ == '__main__':
    main()