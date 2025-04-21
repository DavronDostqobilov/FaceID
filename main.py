from flask import Flask, request, jsonify
import face_recognition
import cv2
import numpy as np
import requests

app = Flask(__name__)

def download_image(url):
    response = requests.get(url)
    if response.status_code != 200:
        return None
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)

@app.route('/')
def hello():
    return jsonify({"Hello": "World"})

@app.route('/face-compare', methods=['POST'])
def face_compare():
    face1_url = request.form.get('face1') # passport img
    face2_url = request.form.get('face2')

    if not face1_url or not face2_url:
        return jsonify({"error": "Iltimos, ikkala URL ni ham yuboring!"}), 400

    # Download and decode images
    img1 = download_image(face1_url)
    img2 = download_image(face2_url)

    if img1 is None or img2 is None:
        return jsonify({"error": "Rasmlarni olishda xatolik yuz berdi"}), 400

    # Check face2 quality
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    if fm < 100:
        return jsonify({'error': 'Yuz sifati past, iltimos aniqroq rasm yuboring!'}), 400

    # Convert images to RGB
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Encode faces
    enc1 = face_recognition.face_encodings(img1_rgb)
    enc2 = face_recognition.face_encodings(img2_rgb)

    if not enc1:
        return jsonify({'error': 'FACE1 dan yuz topilmadi!'}), 400
    if not enc2:
        return jsonify({'error': 'FACE2 dan yuz topilmadi!'}), 400

    # Compare faces
    result = face_recognition.compare_faces([enc1[0]], enc2[0])
    confidence = face_recognition.face_distance([enc1[0]], enc2[0])[0]

    return jsonify({
        "match": bool(result[0]),
        "confidence": round(float(confidence), 4)
    })

if __name__ == '__main__':
    app.run(debug=True)
