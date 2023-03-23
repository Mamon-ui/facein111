import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

path = os.getcwd() + '/images'
images = []
classNames = []
encodeListKnown = []


def encode_images():
    global encodeListKnown, classNames, images
    encodeListKnown = []
    images = []
    classNames = []
    myList = os.listdir(path)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeListKnown.append(encode)
    print("encoding comp")


encode_images()
print(len(encodeListKnown))


@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == "GET":
        return jsonify({"method": "GET"})

    try:
        print(len(encodeListKnown))
        image = request.files['image']
        image_file = image.read()
        np_image = np.frombuffer(image_file, np.uint8)
        img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(img)
        encodesCurFrame = face_recognition.face_encodings(img, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                response = jsonify({'id': name})
                print(name)
                response.headers.add("Access-Control-Allow-Origin", "*")
                response.headers.add("Access-Control-Allow-Headers", "*")
                response.headers.add("Access-Control-Allow-Methods", "*")
                return response
        response = jsonify({'error': "Image not recognized"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route("/add", methods=["POST"])
def add():
    try:
        id = request.form.get('id')
        if id in classNames:
            raise ValueError('Image with same ID number already exists')

        image = request.files['image']
        image1 = image.read()
        np_image = np.frombuffer(image1, np.uint8)
        img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        tid = str(id)
        cv2.imwrite(os.path.join(path, tid + ".jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        classNames.append(tid)
        encode_images()
        print("encoding comp for new image")
        print(len(encodeListKnown))
        response = jsonify({'status': 'success'})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "*")
        response.headers.add("Access-Control-Allow-Methods", "*")
        return response

    except Exception as e:
        return jsonify({"Error => ": str(e)})


@app.route("/edit", methods=["POST"])
def edit():
    global encodeListKnown
    try:
        image = request.files['image']
        id = request.form.get('id')
        tid = str(id)

        if tid not in classNames:
            return jsonify({'error': f"Employee with id {id} does not exist"})

        # delete the old image file
        os.remove(os.path.join(path, tid + ".jpg"))
        classNames.remove(tid)
        encodeListKnown = []

        # save the new image file
        image1 = image.read()
        np_image = np.frombuffer(image1, np.uint8)
        img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cv2.imwrite(os.path.join(path, tid + ".jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # re-encode all the images
        encode_images()
        print("encoding comp for new image")
        return jsonify({'status': 'success'})

    except Exception as e:
        return jsonify({"Error => ": str(e)})


if __name__ == '__main__':
    encode_images()
    app.run(host='0.0.0.0', port=8080)