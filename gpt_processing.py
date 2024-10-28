import ast
import os, cv2
from openai import OpenAI
from main import predict

API_KEY = 'YOUR OPENAI API KEY'
os.environ["OPENAI_API_KEY"] = API_KEY
client = OpenAI()

results = predict('video.MP4')
label_results = []
for r in results:
    frame_results = []
    boxes = r.boxes.xyxy.cpu().numpy()
    labels = r.boxes.cls.cpu().numpy()
    img_width = r.orig_img.shape[1]
    img_height = r.orig_img.shape[0]

    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        normalized_box = (
            float(xmin / img_width),
            float(ymin / img_height),
            float(xmax / img_width),
            float(ymax / img_height)
        )
        frame_results.append((normalized_box, int(label)))

    label_results.append(frame_results)

data_summary = "\n".join([f"Frame {i}: {x}" for i, x in enumerate(label_results[0::5])])
print(data_summary)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": """You are a voice assistant for a car driver, the input is the actual data
                        collected by a driving camera, and the camera is always facing front, please provide the key information for
                        the driver to help him navigate and avoid potential danger"""
        },
        {
            "role": "user",
            "content":
                """I'm working on processing videos that have predictions from a YOLOv8 model.
                The categories include "car with label 0", "pedestrian with label 1", "truck with label 3" and "biker with label 2".
                These are normalized bounding boxes and corresponding labels for each frame.
                (The first tuple indicates the bounding box coordinates: (xmin, ymin, xmax, ymax), and the second tuple indicates the corresponding label)
                For example:
                A detected car might be represented as:
                [((0.4, 0.5, 0.5, 0.6), 0)] where (0.4, 0.5, 0.5, 0.6) is coordinate of bounding box and 0 is label for car.
                Similar for pedestrian, biker and truck with their corresponding labels 1, 2, 3.
                For each frames from the resulted video, and please do the following.
                Alerting for Detected Objects:
                Emergency levels should be defined as:
                -1: No labels detected.
                1: as long as Car detected.
                2: as long as Truck detected.
                3: as long as Cyclist detected.
                4: as long as People detected.
                Please prioritize description based on following percentage:
                People: 0.45, Cyclist: 0.3, Truck: 0.15, Car: 0.1
                However, please prioritize objects that are close to the bottom one third of the frame.
                Frame Annotation:
                For each frame, e.g. Frame 1, inputted, please give a concise description of all detected objects. Include their specific positions relative to the viewing angle and a cautionary statement.
                Note: Format the result as a python list where each element in list is a dictionary for a frame.
                For instance, result with only one frame should be [{"Frame":1, "Emergency level":4, "Description":"People on your right, several cars far away, slow down!"}]"""
        },
        {
            "role": "user",
            "content": "Here are the frames of video:" + data_summary
        }
    ]
)

description_text = response.choices[0].message.content
print(description_text)

start_index = description_text.index("[")
end_index = description_text.rindex("]") + 1
list_str = description_text[start_index:end_index]
descriptions = ast.literal_eval(list_str)

video = cv2.VideoCapture("output_video.mp4")
frame_shape = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
output_video = cv2.VideoWriter("annotated_output_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, frame_shape)

frame_idx = 0
while True:
    ret, frame = video.read()
    if not ret:
        break

    cur = descriptions[frame_idx // 5] if frame_idx // 5 < len(descriptions) else {'Emergency level': 0, 'Description': ''}
    emergency_level = cur['Emergency level']
    description = cur['Description']


    cv2.putText(frame, f"Emergency Level: {emergency_level}", (10, frame_shape[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
    cv2.putText(frame, description, (10, frame_shape[1] - 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)

    output_video.write(frame)
    frame_idx += 1

video.release()
output_video.release()