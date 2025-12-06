from ultralytics import YOLO

model = YOLO("yolov11n.pt") # load the pre trained model
results = model.train(data = "data.yaml", epochs=100, imgsz=640)

# validate the model
metrics = model.val()
print(metrics.box.map)

# test the model
results = model.predict(source = "test_images/image1.jpg", conf=0.25, save=True)