import cv2
import numpy as np
import torch
import time

model = torch.hub.load('./yolov5', 'custom', path='./weights/ppe_yolo_n.pt',source='local')  # local repo
model.conf = 0.4

cap = cv2.VideoCapture(0)

fps_time = time.time()

while True:

    ret,frame = cap.read()

    frame = cv2.flip(frame,1)

    img_cvt = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    # Inference
    results = model(img_cvt)
    result_np = results.pandas().xyxy[0].to_numpy()

    # print(result_np)

    for box in result_np:
        l,t,r,b = box[:4].astype('int')
        # label_id
        label_id = box[5]
        # confidence
        confidence = box[4]
        # label_name
        label_name = box[6]
        print(f'置信度为{confidence},标签序号为{label_id},标签名称为{label_name}'.format(confidence=confidence,label_id=label_id,label_name=label_name))
        cv2.rectangle(frame,(l,t),(r,b),(0,255,0),5)


    now = time.time()
    fps_text = 1/(now - fps_time)
    fps_time =  now

    cv2.putText(frame,str(round(fps_text,2)),(50,50),cv2.FONT_ITALIC,1,(0,255,0),2)


    cv2.imshow('demo',frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

    


