import cv2
import tensorflow
import keras
from PIL import Image, ImageOps
import numpy as np
import datetime


cam = cv2.VideoCapture(0)  # เปิดกล้อง

# อ่านไฟล์ที่ดาวน์โหลดมา
face_cascade = "haarcascade_frontalface_default.xml"
face_classifier = cv2.CascadeClassifier(face_cascade)
np.set_printoptions(suppress=True)

model = tensorflow.keras.models.load_model('keras_model.h5')

while True:
    check, image_bgr = cam.read()  # รับภาพจากกล้อง

    # show time
    timenow = str(datetime.datetime.now())
    cv2.putText(image_bgr, timenow, (10, 30), 2, 0.8, (0, 0, 0))

    # image_bgr = cv2.flip(image_bgr, 1)
    image_bw = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # จำแนกใบหน้า
    faces = face_classifier.detectMultiScale(image_bw, 1.3, 3)
    image_org = image_bgr.copy()

    for (x, y, w, h) in faces:
        cface_rgb = Image.fromarray(image_rgb[y:y+h, x:x+w])
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = cface_rgb
        image = ImageOps.fit(image, (224, 224), Image.ANTIALIAS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)
        # print(prediction)

        # ใส่เเมส
        if prediction[0][0] > prediction[0][1]:
            cv2.putText(image_bgr, "Please wear a mask!", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(image_bgr, (x, y), (x+w, y+h), (0, 0, 255), 2)
        # ไม่ใส่เเมส
        else:
            cv2.putText(image_bgr, "You are wearing a mask", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(image_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Mask Detection", image_bgr)

    # กด x เมื่อต้องการปิดโปรแกรม
    if (cv2.waitKey(1) & 0xFF == ord("x")):
        break

cam.release()
cv2.destroyAllWindows()
