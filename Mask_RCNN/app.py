import imghdr
import os
from pickle import NONE
from flask import Flask, render_template, request, redirect, url_for, abort, \
    send_from_directory,jsonify,session
from flask_dropzone import Dropzone
from flask_ngrok import run_with_ngrok #comment if not using colab

from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import json
import sys
from flask_dropzone import Dropzone
#for visualizing outputs
import matplotlib.pyplot as plt

#ModelspecificPackages============

ROOT_DIR = os.getcwd()
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize


import keras.backend

K = keras.backend.backend()
if K=='tensorflow':
    keras.backend.set_image_dim_ordering('tf')


#=====================================
#configureFlaskApp
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png']
app.config['UPLOAD_PATH'] = 'uploads'
app.config['MODEL_PATH'] = 'logs/teeth20230201T2159/mask_rcnn_teeth_0238.h5'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.secret_key = "thisisasupersecretkey"  


app.config.update(
DROPZONE_REDIRECT_VIEW='prediction',  # set redirect view
DROPZONE_MAX_FILES=20,
)

dropzone = Dropzone(app)
run_with_ngrok(app) #comment if not using colab
#=======================================

#global inference model=================
class InferenceConfig(Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 12 # 1 Background + 11 teeth classes
    NAME = "teeth"

    DETECTION_MIN_CONFIDENCE = 0.8
    BACKBONE = "resnet101"

config = InferenceConfig()

#initialize maskrcnn model
model = modellib.MaskRCNN(mode="inference", model_dir="./logs", config=config)
#load model weights globally
model.load_weights(app.config['MODEL_PATH'], by_name=True)
model.keras_model._make_predict_function()
class_names = ['BG','kirik', 'bosluk', 'dolgu', 'kanal_tedavi', 'implant', 'curuk', 'yirmilik_dis', 'tel', 'saglikli', 'plak', 'dis_tasi']
id_category = [0,1,2,3,4,5,6,7,8,9,10,11]

#=========================================



def validate_image(stream):
    header = stream.read(512)
    stream.seek(0)
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')

#this function is used to predict on the uploaded image
def predict_on_image(uploaded_file):
    global model
    img = Image.open(uploaded_file)
    img = np.array(img) 

    results = model.detect([img], verbose=0)
    fig, ax = plt.subplots(figsize=(16, 16)) 
    r = results[0]
    #visualize results and save them to file
    visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
                                class_names, r['scores'],figsize=(16,16), ax=ax)
    fig.savefig('static/prediction.png',bbox_inches='tight')   # save the figure to file
    plt.close(fig)  
    response =[] 
    for p,scr in zip(results[0]['class_ids'],results[0]['scores']):
        response.append({"teeth":class_names[p],"score":str(scr)})
    return response

@app.errorhandler(413)
def too_large(e):
    return "File is too large", 413

@app.route('/', methods=['GET','POST'])
def upload_files():
     if request.method == "POST":
        uploaded_file = request.files['file']
        filename = secure_filename(uploaded_file.filename)
        if filename != '':
            file_ext = os.path.splitext(filename)[1]
            if file_ext not in app.config['UPLOAD_EXTENSIONS'] or \
                    file_ext != validate_image(uploaded_file.stream):
                return "Invalid image", 400
            response = predict_on_image(uploaded_file)
            print(response)
            session["response"]=response
            return render_template("prediction.html",jsonfile = session["response"])
           
     else:
        return render_template('index.html')


@app.route('/prediction',methods=['GET','POST'])
def prediction():
    if request.method == "GET":
        return render_template("prediction.html",jsonfile = session["response"])
    else:
        uploaded_file = request.files['file']
        filename = secure_filename(uploaded_file.filename)
        if filename != '':
            file_ext = os.path.splitext(filename)[1]
            if file_ext not in app.config['UPLOAD_EXTENSIONS'] or \
                    file_ext != validate_image(uploaded_file.stream):
                return "Invalid image", 400
            response = predict_on_image(uploaded_file)
            session["response"]=response
        return render_template("prediction.html",jsonfile = session["response"])



if __name__=='__main__':
    app.run()