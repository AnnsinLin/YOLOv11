import warnings
 
warnings.filterwarnings('ignore')


from ultralytics import YOLO
# Load a model
### model = YOLO('yolo11n.pt')  # build a new model from YAML
# model = YOLO('yolov8s-pose.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='fengdian.yaml', epochs=200, batch=8, device='cpu')