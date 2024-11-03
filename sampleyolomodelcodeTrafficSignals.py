from roboflow import Roboflow
rf = Roboflow(api_key="1MrSAw6ydgH3PZZFITLz")
project = rf.workspace("vo-vy").project("trafficsigns-p363x")
version = project.version(1)
dataset = version.download("yolov8")