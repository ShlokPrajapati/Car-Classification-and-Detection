import cv2
import math
from ultralytics import YOLO

cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
cap=cv2.VideoCapture('video22.mp4')

model=YOLO('yolov8n.pt')
mask=cv2.imread('mask.png')
classnames=["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
             "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "con", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball glove", "skateboard", "surfboard", "tennis racket",
                 "bottle", "wine glass", "cup", "fork", "knife", "spoon", "boal", "banana", "apple", "sandwich", "orange", "broccoli",
                   "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "Laptop", "mouse", "remote",
                     "keyboard", "cell phone", "microsave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors","teddy bear", "hair drier",
                       "toothbrush"]


def cornerRect(img, bbox, l=30, t=5, rt=1,
               colorR=(255, 0, 255), colorC=(0, 255, 0)):
    """
    :param img: Image to draw on.
    :param bbox: Bounding box [x, y, w, h]
    :param l: length of the corner line
    :param t: thickness of the corner line
    :param rt: thickness of the rectangle
    :param colorR: Color of the Rectangle
    :param colorC: Color of the Corners
    :return:
    """
    x, y, w, h = bbox
    x1, y1 = x + w, y + h
    if rt != 0:
        cv2.rectangle(img, bbox, colorR, rt)
    # Top Left  x,y
    cv2.line(img, (x, y), (x + l, y), colorC, t)
    cv2.line(img, (x, y), (x, y + l), colorC, t)
    # Top Right  x1,y
    cv2.line(img, (x1, y), (x1 - l, y), colorC, t)
    cv2.line(img, (x1, y), (x1, y + l), colorC, t)
    # Bottom Left  x,y1
    cv2.line(img, (x, y1), (x + l, y1), colorC, t)
    cv2.line(img, (x, y1), (x, y1 - l), colorC, t)
    # Bottom Right  x1,y1
    cv2.line(img, (x1, y1), (x1 - l, y1), colorC, t)
    cv2.line(img, (x1, y1), (x1, y1 - l), colorC, t)

    return img


def putTextRect(img, text, pos, scale=3, thickness=3, colorT=(255, 255, 255),
                colorR=(255, 0, 255), font=cv2.FONT_HERSHEY_PLAIN,
                offset=10, border=None, colorB=(0, 255, 0)):
    """
    Creates Text with Rectangle Background
    :param img: Image to put text rect on
    :param text: Text inside the rect
    :param pos: Starting position of the rect x1,y1
    :param scale: Scale of the text
    :param thickness: Thickness of the text
    :param colorT: Color of the Text
    :param colorR: Color of the Rectangle
    :param font: Font used. Must be cv2.FONT....
    :param offset: Clearance around the text
    :param border: Outline around the rect
    :param colorB: Color of the outline
    :return: image, rect (x1,y1,x2,y2)
    """
    ox, oy = pos
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)

    x1, y1, x2, y2 = ox - offset, oy + offset, ox + w + offset, oy - h - offset

    cv2.rectangle(img, (x1, y1), (x2, y2), colorR, cv2.FILLED)
    if border is not None:
        cv2.rectangle(img, (x1, y1), (x2, y2), colorB, border)
    cv2.putText(img, text, (ox, oy), font, scale, colorT, thickness)

    return img, [x1, y2, x2, y1]

while True:
    success,img=cap.read()
    regionImg=cv2.bitwise_and(img,mask)
    results=model(regionImg,stream=True)
    for r in results:
        boxes=r.boxes
        for box in boxes:
            # Bounding box
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            w,h=x2-x1,y2-y1

            # Confidence

            conf=math.ceil((box.conf[0]*100)) / 100

            # Classname
            cls=int(box.cls[0])
            currentClass=classnames[cls]
            if (currentClass=='car' or currentClass=='bus' or currentClass=='truck' or currentClass=='motorbike') and conf>0.5:
                cornerRect(img,(x1,y1,w,h),l=10)
                putTextRect(img,f'{currentClass} {conf}',(max(0,x1),max(35,y1)),scale=0.7,thickness=2,offset=3)

    cv2.imshow('Video',img)
    cv2.waitKey(2)