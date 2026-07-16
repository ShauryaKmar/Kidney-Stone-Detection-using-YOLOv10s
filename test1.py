from ultralytics import YOLO

model = YOLO(r"runs\detect\AI_Demo\kidney_stone_test4\weights\best.pt")
model.model.names = {0: "Stone", 1: "Normal Kidney"}
model.save(r"runs\detect\AI_Demo\kidney_stone_test4\weights\best.pt")