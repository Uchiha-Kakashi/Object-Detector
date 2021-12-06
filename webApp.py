from flask import Flask, request, render_template
from werkzeug.utils import redirect
from flask_sqlalchemy import SQLAlchemy

import os
from datetime import datetime

from yolo_custom_predictor import Yolodetector


app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///yolov3.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class Images(db.Model):
    img_ID = db.Column(db.Integer, primary_key = True)
    img_path = db.Column(db.String(100), nullable = True)
    predict_path = db.Column(db.String(100), nullable = True)
    classes = db.Column(db.String(200), nullable = True)
    timestamp = db.Column(db.DateTime, default = datetime.utcnow)

@app.route('/yolov3detector', methods = ['GET', 'POST'])
def homepage():
    if request.method == 'POST':
        imgFile = request.files['image']        
        
        if not imgFile:
            return '<h1> Please upload a valid file!! </h1>'

        Img = Images()
        db.session.add(Img)
        db.session.flush()
        
        img_path = saveImg(imgFile, Img.img_ID)
        
        UplImg = Images.query.get(Img.img_ID)
        UplImg.img_path = img_path
        
        if imgFile:
            imgClass = Yolodetector(img_path)
            results, predictPath = imgClass.detector()
        
        classString = ''
        if len(results) > 0:
            for obj in results:
                classString += obj + " "
        else:
            classString = 'None'

        UplImg.classes = classString
        UplImg.predict_path = predictPath

        db.session.commit()

        # return '<h1> Uploaded successfully at {}<h1>'.format(img_path)
        return redirect('/showprediction/{}'.format(UplImg.img_ID))
        

    elif request.method == 'GET':
        return render_template('upload_img.html')

@app.route('/showprediction/<int:id>', methods = ['GET'])
def showImages(id):

    ImgPredicted = Images.query.get(int(id))

    predictPath = ImgPredicted.predict_path
    classes = ImgPredicted.classes

    if not os.path.isfile(predictPath):

        return '<h1> The file is not present anymore. Try uploading once more!!! </h1>'

    return render_template('showImages.html', ImgPath = '/' + predictPath, classes = classes)

@app.route('/viewhistory', methods = ['GET'])
def History():
    ImagesTable = Images.query.all()
    return render_template('history.html', Images = ImagesTable)

@app.route('/delete/<int:id>')
def Deleterecord(id):

    DelImg = Images.query.get(id)

    #removing the image files from file Directory
    os.remove(DelImg.predict_path)
    os.remove(DelImg.img_path)

    db.session.delete(DelImg)
    db.session.commit()

    return redirect('/viewhistory')


def saveImg(imgFile, id):

    tempPath = 'static/Images/Uploaded/{}.jpg'.format(id)

    imgFile.save(tempPath)

    return tempPath


if __name__ == '__main__':
    app.run(port=80)