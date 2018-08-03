import cv2 as cv
import dlib
import keras.backend as K
import numpy as np

from resnet_101 import resnet101_model


def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return x, y, w, h


def load_emotion_model(model_path):
    img_width, img_height = 224, 224
    num_channels = 3
    num_classes = 7

    emotion_model = resnet101_model(img_height, img_width, num_channels, num_classes)
    emotion_model.load_weights(model_path, by_name=True)
    return emotion_model


def draw_bounding_box(image, coordinates, color):
    x, y, w, h = coordinates
    cv.rectangle(image, (x, y), (x + w, y + h), color, 2)


def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return x - x_off, x + width + x_off, y - y_off, y + height + y_off


def draw_text(image, coordinates, text, color, x_offset=0, y_offset=0,
              font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv.putText(image, text, (x + x_offset, y + y_offset),
               cv.FONT_HERSHEY_SIMPLEX,
               font_scale, color, thickness, cv.LINE_AA)


def get_color(emotion, prob):
    if emotion.lower() == 'angry':
        color = prob * np.asarray((0, 0, 255))
    elif emotion.lower() == 'sad':
        color = prob * np.asarray((255, 0, 0))
    elif emotion.lower() == 'happy':
        color = prob * np.asarray((0, 255, 255))
    elif emotion.lower() == 'surprise':
        color = prob * np.asarray((255, 255, 0))
    elif emotion.lower() == 'fear':
        color = prob * np.asarray((255, 255, 255))
    elif emotion.lower() == 'disgust':
        color = prob * np.asarray((255, 0, 255))
    else:
        color = prob * np.asarray((0, 255, 0))
    return color


def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)


def predict(filename):
    img_width, img_height = 224, 224

    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    # class_names = ['愤怒', '厌恶', '恐惧', '高兴', '悲伤', '惊讶', '无表情']

    detector = dlib.get_frontal_face_detector()
    emotion_model = load_emotion_model('models/model.best.hdf5')

    image = cv.imread(filename)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.cvtColor(gray, cv.COLOR_GRAY2RGB)
    faces = detector(gray, 1)

    for rect in faces:
        (x, y, w, h) = rect_to_bb(rect)
        x1, x2, y1, y2 = apply_offsets((x, y, w, h), (10, 10))
        gray_face = gray[y1:y2, x1:x2]
        gray_face = cv.resize(gray_face, (img_height, img_width))
        gray_face = np.expand_dims(gray_face, 0)
        preds = emotion_model.predict(gray_face)
        prob = np.max(preds)
        class_id = np.argmax(preds)
        emotion = class_names[class_id]
        color = get_color(emotion, prob)
        draw_bounding_box(image=image, coordinates=(x1, y1, x2 - x1, y2 - y1), color=color)
        draw_text(image=image, coordinates=(x1, y1, x2 - x1, y2 - y1), color=color, text=emotion)

    cv.imwrite('images/output.png', image)

    K.clear_session()
    return image
