# Yolov8_Driving_Assistant
- This project employs a trained YOLOv8 object detection model to identify and annotate objects in video footage. By integrating OpenAI's GPT model, the system generates real-time emergency alerts and descriptive messages based on the detected objects in each frame. This functionality is designed to effectively convey critical information to drivers, enhancing situational awareness and promoting safer driving practices.

https://github.com/user-attachments/assets/6687525b-7e7e-4f12-a83f-83cae6b6bb3a

## Training Process
### Dataset: KITTI Object Detection and COCO
- Preprocessing: The dataset was carefully curated by selecting specific object classes (cars, pedestrians, cyclists, trucks) and converting the annotations into the YOLOv8 training format.
- Train-Val-Test Split: The dataset was divided into training, validation, and testing sets with a ratio of 8:1:1.
- Model Training: Utilized a pre-trained YOLOv8 model from the Ultralytics package to fine-tune on the prepared dataset.
- Performance Metrics: The final accuracy achieved on the validation set was as follows:
  Cars: 95%
  People: 82%
  Cyclists: 86%
  Trucks: 97%
## GPT Process
- Label Processing: Leveraged the OpenAI Chat API to analyze the labels obtained from predictions made on video inputs. A carefully crafted prompt was provided to the language model, specifying criteria for emergency levels and object descriptions.
- Real-time Alerts: Generated scripts from the language model were utilized to annotate the video with emergency levels and descriptive alerts, effectively communicating potential hazards to drivers in real-time. 
