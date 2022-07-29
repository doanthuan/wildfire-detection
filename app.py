from __future__ import division
import os
from flask import Flask, request, redirect, url_for, render_template, flash, jsonify, send_from_directory, Response
from werkzeug.utils import secure_filename
from datetime import datetime
from threadcamera import ThreadedCamera
from utils import *
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img , img_to_array
import tensorflow as tf


app = Flask(__name__)




app.config['UPLOAD_FOLDER'] = 'uploads'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(BASE_DIR , 'models/best_model.h5'))

ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png' , 'jfif'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

classes = ['fire' ,'nofire']
img_width = 224
img_height = 224
image_size=(img_width, img_height)


def predict(filename):
    img = tf.keras.utils.load_img(
        filename, target_size=(img_height, img_width)
    )
    return predict_img(img, False)

def predict_img(img, resize=True):
    if resize:
        img = tf.image.resize(img, (img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    predictions = tf.where(predictions > 0, 1, 0)
    # print(predictions)
    # score = tf.nn.softmax(predictions[0])

    label = classes[predictions[0][0]]
    #score = 100 * np.max(score)
    return label

@app.route('/', methods=['GET'])
def index(name=None):
    return render_template("upload.html")

@app.route('/upload', methods=['GET','POST'])
def upload(name=None):
    if request.method == 'POST':
        # check if the post request has the file part
        if 'upload_file' not in request.files:
            return 'No file part'
        file = request.files['upload_file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return 'No selected file'

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        #check if the file is video
        split_tup = os.path.splitext(filepath)
        file_ext = split_tup[1]
        if file_ext == ".mp4":
            return redirect(url_for('video_feed', name=filename))
        else:
            #image classify
            label = predict(filepath)
            #display result
            return render_template("result.html", filename=filename, label=label)
    else:
        return render_template('upload.html')

def parse_video(filename):

    # Read from video file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    # cap = cv2.VideoCapture(filepath)
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    # out = cv2.VideoWriter(path,fourcc, 20, (460,360))
    cap = ThreadedCamera(filepath)

    # define a box of Roid
    frame_number = 0
    objectID = 0
    frame_number = 0
    frame = None
    prev_frame = None
    while True:
        #start_time = time.time()
        frame = cap.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        if prev_frame is not None:
            # --- take the absolute difference of the images ---
            res = cv2.absdiff(frame, prev_frame)
            # --- convert the result to integer type ---
            res = res.astype(np.uint8)
            # --- find percentage difference based on number of pixels that are not zero ---
            percentage = (np.count_nonzero(res) * 100) / res.size

            if (percentage>10):
                label = predict_img(frame)
                print(label)
        else:
            label = predict_img(frame)
            print(label)

        prev_frame = frame
        frame_number = frame_number + 1

        cv2.putText(frame, label.upper(), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # yield the output frame in the byte format
        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        if not flag:
            continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

    #cap.release()

@app.route("/video-feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
    filename = request.args['name']  
    # counterpart for url_for()
    return Response(parse_video(filename), mimetype = "multipart/x-mixed-replace; boundary=frame")

# Custom static data
@app.route('/cdn/<path:filename>')
def custom_static(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == "__main__":
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True, host='0.0.0.0', port=5001)