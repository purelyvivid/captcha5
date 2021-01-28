# -*- coding: utf-8 -*-
from flask import Flask, jsonify, request, redirect, url_for
import os
import socket
from loadimg import write_img,load_save_img
from detect import detect
import magic
from datetime import datetime

from flask_uploads import UploadSet, configure_uploads, IMAGES,\
 patch_request_class





app = Flask(__name__)
app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd()  # 文件储存地址

photos = UploadSet('photos', IMAGES, default_dest=lambda x: 'photos')
configure_uploads(app, photos)
patch_request_class(app)  # 文件大小限制，默认为16MB

bank_name = "*銀" # modified

html = f'''
    <!DOCTYPE html>
    <title>{bank_name}驗證碼</title>
    <h1>{bank_name}驗證碼</h1>
    <h2>Upload Image</h2>
    <form method=post enctype=multipart/form-data>
         <input type=file name=img_file>
         <input type=submit value=預測>
    </form>
    <h2>From Image URL</h2>
    <form method=post enctype=multipart/form-data>
         URL:<input type=text name=img_file_pth>
         <input type=submit value=預測>
    </form>
    <hr>
    '''


def get_file_url(request):
    filename = photos.save(request.files['img_file'], "static")
    file_url = photos.url(filename)
    return file_url 

def write_result_log(file_url, result_txt, method, source):
    datetime_text = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    with open("result_log.csv", "a") as f:
        f.write(f"{datetime_text},{file_url},{result_txt},{method},{source}\n") 

def get_result(request):
    if ('img_file_base64' in request.form) and (request.form['img_file_base64']!=""):
        method = 'img_file_base64'
        file_url = 'tmp.png'
        write_img(file_url, request.form['img_file_base64']) # save tmp img
        imgarr = load_save_img(file_url) # save tmp img

    elif ('img_file_pth' in request.form) and (request.form['img_file_pth']!=""):
        method = 'img_load_from_path'
        file_url = request.form['img_file_pth'] # remote_file_url
        imgarr = load_save_img(file_url) # save tmp img


    elif 'img_file' in request.files:
        method = 'img_file'
        file_url = get_file_url(request) #local_file_url  
        imgarr = load_save_img(file_url) # save tmp img

    else:
        file_url, result_txt, method  = None, None, "No class"
    
    result_txt = detect()  # modified # 用"tmp.png"去預測

    return file_url, result_txt, method


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file_url, result_txt, method = get_result(request)
        if not file_url is None:
            write_result_log(file_url, result_txt, method, "web_page")
            return html + '<br><img src=' + file_url + '>' + '<br>預測結果: <B>' + result_txt + '</B>'
    return html


@app.route("/predict", methods=['POST'])
def predict():
    file_url, result_txt, method = get_result(request)
    if not file_url is None:
        write_result_log(file_url, result_txt, method, "api_predict")
        return jsonify({'method':method, 'prediction': result_txt}) 
    return jsonify({'method':method, 'prediction': '<Error>'}) # ex. {'prediction':[0]}


if __name__ == "__main__":
    #app.run('localhost', 5000)
    app.run('0.0.0.0', 5000, debug=True)
