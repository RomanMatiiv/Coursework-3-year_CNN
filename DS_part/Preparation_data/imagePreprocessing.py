import cv2 as cv
import numpy as np
#https://github.com/ageitgey/face_recognition - from here (need dlib also)
import face_recognition


def find_face_cv(image, path_to_face_cascade, type_input="image",debug=False):
    """
        Находит лицо (координаты)
    """
    # Загрузка изображения
    if type_input == "image":
        img = cv.imread(image)
    elif type_input == "np.array" or "array":
        img = image
    else:
        raise ValueError("type_input: image/np.array")

    # Поиск лица
    try:
        face_cascade = cv.CascadeClassifier(path_to_face_cascade)
        face = face_cascade.detectMultiScale(img, 1.1, 5)
    except Exception:
        raise Exception("ОШИБКА!!! Либо путь к haarcascade_frontalface_default.xml указан не выерно или же его там нет")

    # Вернуть нужно только 1 лицо
    if len(face) == 0:
        raise ValueError("Face not found")
    # if len(face) > 1:
    #     face = face[0]
    face = face[0]

    if debug:
        print(face)
    # Извлекаю координаты лица (цифры подобраны имперически чтобы захcватывать большую область)
    # for (x, y, w, h) in face:
    #     x1, x2 = x-20, x+w+30
    #     y1, y2 = y-45, y+h+30

    x, y, w, h =face
    # x=face[0]
    # y=face[1]
    # w=face[2]
    # h=face[3]
    x1, x2 = x - 20, x + w + 30
    y1, y2 = y - 45, y + h + 30

    return (x1, x2 , y1 ,y2)


def find_face(image, type_input="image", model="cnn"):
    """
    Находит лицо (координаты)
    """
    # Загрузка изображения
    if type_input == "image":
        img = cv.imread(image,0)
    elif type_input == "np.array" or "array":
        img = image
    else:
        raise ValueError("type_input: image/np.array")

    # Определение местоположения лица
    if model == "cnn":
        # желательно использовать GPU версию метода а не эту
        face_locations = face_recognition.face_locations(img, number_of_times_to_upsample=0, model="cnn")
    elif model == "hog":
        face_locations = face_recognition.face_locations(img)
    else:
        raise ValueError("cnn or hog")

    # Извлекаю координаты только 1 лица
    if len(face_locations) == 0:
        raise ValueError("Face not found")
    else:
        y1, x2, y2, x1 = face_locations[0]

    return (x1, x2, y1, y2)


def angle_rotation_cv(image, path_to_eye_cascade=None, type_input="image"):
    """
        Вычисляет угол, на который нужно повернуть изображение, чтобы глаза были горизонтально
    """

    # Загрузка изображения
    if type_input == "image":
        img = cv.imread(image)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGBA)
    elif type_input == "np.array" or "array":
        img = image
    else:
        raise ValueError("type_input possible values are: image/np.array")

    # Поиск глаз
    if path_to_eye_cascade == None:
        path_to_eye_cascade = "haarcascade_eye.xml"
    eye_cascade = cv.CascadeClassifier(path_to_eye_cascade)
    try:
        eyes = eye_cascade.detectMultiScale(img, 1.1, 2)  # Параметры именно такие
    except Exception:
        raise Exception("ОШИБКА!!! Либо путь к haarcascade_eye.xml указан не выерно или же его там нет")

    if len(eyes) < 2 or len(eyes) > 4:
        raise ValueError("Не удалось корректно обработать фото")

    center = []
    for (ex, ey, ew, eh) in eyes[:2]:
        xc = int(np.mean([ex, ex + ew]))
        yc = int(np.mean([ey, ey + eh]))
        center.append([xc, yc])

    center = np.array(center)

    min_x = np.argmin(center, axis=0)
    max_x = np.argmax(center, axis=0)

    a = center[:, 1].max() - center[:, 1].min()
    b = center[:, 0].max() - center[:, 0].min()
    c = np.sqrt(a ** 2 + b ** 2)

    rad = np.arcsin(a / c)
    degree = np.rad2deg(rad)

    if center[min_x[0]][1] >= center[max_x[0]][1]:
        degree = -degree
    return degree


def angle_rotation(image, type_input="image"):
    """
    Вычисляет угол, на который нужно повернуть изображение, чтобы глаза были горизонтально
    """
    # Загрузка изображения
    if type_input == "image":
        img = cv.imread(image)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    elif type_input == "np.array" or "array":
        img = image
    else:
        raise ValueError("type_input possible values are: image/np.array")

    # Поиск глаз
    # Только для 1 человека
    face_landmarks_list = face_recognition.face_landmarks(img)[0]
    left_eye = np.mean(face_landmarks_list['left_eye'], axis=0)
    right_eye = np.mean(face_landmarks_list['right_eye'], axis=0)

    if left_eye[1] > right_eye[1]:
        a = left_eye[1]-right_eye[1]
    else:
        a = right_eye[1]-left_eye[1]

    if left_eye[0] > right_eye[0]:
        b = left_eye[0]-right_eye[0]
    else:
        b = right_eye[0]-left_eye[0]

    c = np.sqrt(a ** 2 + b ** 2)

    rad = np.arcsin(a / c)
    degree = np.rad2deg(rad)

    if right_eye[0] >= left_eye[0]:
        degree = -degree
    return degree


def rotate_image(image, degree, type_input="image"):
    """
    Поворачивает изображение так чтобы глаза были горизонтально
    """

    # Загрузка изображения
    if type_input == "image":
        img = cv.imread(image)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    elif type_input == "np.array" or "array":
        img = image
    else:
        raise ValueError("type_input possible values are: image/np.array")

    # Поворачивает
    rows, cols = img.shape[:2]
    m = cv.getRotationMatrix2D((cols / 2, rows / 2), degree, 1)
    img = cv.warpAffine(img, m, (cols, rows))
    return img


def crop_cv(image, degree, type_input="image", path_to_eye_cascade=None):
    '''
        Обрезает изображение так, чтобы галза были по центру
    '''
    # Загрузка изображения
    if type_input == "image":
        img = cv.imread(image)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGBA)
    elif type_input == "np.array" or "array":
        img = image
    else:
        raise ValueError("type_input possible values are: image/np.array")

    # Поворачивает
    rows, cols = img.shape[:2]
    m = cv.getRotationMatrix2D((cols / 2, rows / 2), degree, 1)
    img = cv.warpAffine(img, m, (cols, rows))

    # Центрует
    if path_to_eye_cascade == None:
        path_to_eye_cascade = "haarcascade_eye.xml"
    eye_cascade = cv.CascadeClassifier(path_to_eye_cascade)
    try:
        eyes = eye_cascade.detectMultiScale(img)  # Параметры именно такие(дефолтные)
    except Exception:
        raise Exception("ОШИБКА!!! Либо путь к haarcascade_eye.xml указан не выерно или же его там нет")

    if len(eyes) < 2 or len(eyes) > 4:
        raise ValueError("Не удалось корректно обработать фото")

    center = []
    for (ex, ey, ew, eh) in eyes[:2]:
        xc = int(np.mean([ex, ex + ew]))
        yc = int(np.mean([ey, ey + eh]))
        center.append([xc, yc])

    center = np.array(center)

    # По горизонтали
    x_min = 0
    x_max = img.shape[1]
    x_l, x_r = np.sort(center[:, 0])

    left = x_l - x_min
    right = x_max - x_r

    if left > right:
        img = img[:, :x_r + left]

    if left < right:
        img = img[:, x_l - left:]

    # По вертикали
    y_min = 0
    y_max = img.shape[0]
    y = np.sort(center[:, 1])[0]

    hi = y - y_min
    low = y_max - y

    if hi > low:
        img = img[y - low:, :]

    if hi < low:
        img = img[:y + hi, :]

    return img


def crop(image, degree, type_input="image"):
    """
    Обрезает изображение так, чтобы галза были по центру
    """
    # Загрузка изображения
    if type_input == "image":
        img = cv.imread(image)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    elif type_input == "np.array" or "array":
        img = image
    else:
        raise ValueError("type_input possible values are: image/np.array")

    # Поворачивает
    if degree != 0:
        rows, cols = img.shape[:2]
        m = cv.getRotationMatrix2D((cols / 2, rows / 2), degree, 1)
        img = cv.warpAffine(img, m, (cols, rows))

    # Центрует

    # Находит глаза
    try:
        face_landmarks_list = face_recognition.face_landmarks(img)[0]
    except IndexError:
        raise IndexError("Глаза не найдены")
    left_eye = np.mean(face_landmarks_list['left_eye'], axis=0)
    right_eye = np.mean(face_landmarks_list['right_eye'], axis=0)

    # По горизонтали
    x_min = 0
    x_max = img.shape[1]
    x_l, x_r = int(left_eye[0]), int(right_eye[0])

    left = x_l - x_min
    right = x_max - x_r

    if left > right:
        img = img[:, x_l - right:]

    if left < right:
        img = img[:, :x_r + left]

    # По вертикали
    y_min = 0
    y_max = img.shape[0]
    y = int(left_eye[1])

    hi = y - y_min
    low = y_max - y

    if hi > low:
        img = img[y - low:, :]

    if hi < low:
        img = img[:y + hi, :]

    return img


def alignment_cv(image, path_to_face_cascade, path_to_eye_cascade=None, type_input="image"):
    """
    Делает тоже самое что и alignment но только с помошью opencv (качество по хуже)
    Args:
        image: input image
        path_to_face_cascade: путь до файла haarcascade_frontalface_default.xml" (с помощью него opencv лицо и находит)
        path_to_eye_cascade: путь до файла haarcascade_eye.xml (с помощью него opencv глаза и находит)
        type_input: type of image ("image" or "array")

    Returns:  prepare image
    """
    # Загрузка изображения
    if type_input == "image":
        img = cv.imread(image,0)
        # img = cv.cvtColor(img, cv.COLOR_BGR2RGBA)
    elif type_input == "np.array" or "array":
        img = image
    else:
        raise ValueError("type_input possible values are: image/np.array")

    x1, x2, y1, y2 = find_face_cv(img, path_to_face_cascade, type_input="array")
    crop_img = img[y1:y2, x1:x2]
    degree = angle_rotation_cv(crop_img, path_to_eye_cascade, type_input="array")
    # rotate_img = rotate_image(img, degree, type_input="array") хуже работает сука
    rotate_img = rotate_image(image, degree)

    x1, x2, y1, y2 = find_face_cv(rotate_img, path_to_face_cascade, type_input="array")
    crop_img = rotate_img[y1:y2, x1:x2]
    al_crop_img = crop_cv(crop_img, 0, type_input="array", path_to_eye_cascade=path_to_eye_cascade)

    return al_crop_img


def alignment(image, type_input="image", model="cnn"):
    """
    Принимает фото с человеком и возвращает фото с лицом по центру(по 2 осям)
    Args:
        image: input image
        type_input: type of image ("image" or "array")
        model: "cnn" or "hog" cnn-great accuracy but not fast

    Returns: prepare image
    """
    # Загрузка изображения
    if type_input == "image":
        img = cv.imread(image)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    elif type_input == "np.array" or "array":
        img = image
    else:
        raise ValueError("type_input possible values are: image/np.array")

    x1, x2, y1, y2 = find_face(img, type_input="array", model=model)
    crop_img = img[y1:y2, x1:x2]
    degree = angle_rotation(crop_img, type_input="array")
    rotate_img = rotate_image(img, degree, type_input="array")

    x1, x2, y1, y2 = find_face(rotate_img, type_input="array", model=model)
    crop_img = rotate_img[y1-60:y2+40, x1-40:x2+40]  # Расщираю область чтоб при центровке глаз обрезать
    al_crop_img = crop(crop_img, 0, type_input="array")

    return al_crop_img


def resizeFace(distanse):
    """
    Приводит все лица к одному размеру(маштабу)
    Args:
        distanse: желаемое расстояние между глазами (поидее по нему можно сделать все фото одинаковыми)
    """
    return None
