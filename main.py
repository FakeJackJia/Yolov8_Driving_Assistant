import multiprocessing
import cv2
from PIL import Image
from ultralytics import YOLO
import torch
# print(torch.backends.cudnn.enabled)
# print(torch.cuda.is_available())

torch.cuda.set_device(0)
device = torch.device("cuda")

def train():
    model = YOLO('yolov8n.pt')
    model.to(device=device)
    model.train(data='dataset/data.yaml', epochs=50, batch=16)

def predict(img_path):
    model = YOLO('best.pt')
    return model.predict(img_path)

def show(results):
    for i, r in enumerate(results):
        im_bgr = r.plot()
        im_rgb = Image.fromarray(im_bgr[..., ::-1])
        r.show()
        r.save(filename=f"results{i}.jpg")

def save_video(results, output_path, fps=30):
    if results:
        height, width = results[0].plot().shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for r in results:
            im_bgr = r.plot()
            out.write(im_bgr)

        out.release()

if __name__ == '__main__':
    # multiprocessing.freeze_support()
    # train()

    results = predict('video.MP4')
    save_video(results, 'output_video.mp4')