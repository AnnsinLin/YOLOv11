from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(model=r'D:\qq\ultralytics-main\runs\detect\train5\weights\best.pt')
    model.predict(source=r'D:\qq\ultralytics-main\datasets\fengdian\images\train',
                  save=True,
                  show_labels=False,
                  show_conf=False,
                  line_width=2
                  )