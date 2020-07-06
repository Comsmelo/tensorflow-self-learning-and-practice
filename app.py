import flask
import werkzeug 
import os 
import execute
import getConfig
import requests
import pickle
from flask import request, jsonify
import numpy as np
from PIL import Image

global secure_filename

gConfig = {}
gConfig = getConfig.get_config(config_file = 'config.ini')

# 实例化一个Flask应用
app = flask.Flask('imgClassifierWeb')

def CNN_predict():
    # 从原型文件上找到待预测图片的基本信息
    file_ = gConfig['dataset_path'] + 'batches.meta'
    patch_bin_file = open(file_, 'rb')
    label_names_dict = pickle.load(patch_bin_file)['label_names']
    
    global secure_filename
    # 从本地取出要分类的图片
    img = Image.open(os.path.join(app.root_path, secure_filename ))
    r,g,b = img.split()
    r_arr = np.array(r)
    g_arr = np.array(g)
    b_arr = np.array(b)
    # 拼接
    img = np.concatenate((r_arr, g_arr, b_arr))
    image = img.reshape([1, 32, 32, 3])/255
    # 预测
    predicted_class = execute.predict(image)
    predicted_class = label_names_dict[predicted_class]
    # 将返回结果和页面模版渲染出来
    return flask.render_template(template_name_or_list = 'prediction_result.html', predicted_class = predicted_class)

app.add_url_rule(rule = '/predict/', endpoint='predict', view_func = CNN_predict)


def upload_image():
    global secure_filename
    try:    
        if flask.request.method == 'POST':
            # 获取需要分类的图片
            img_file = flask.request.files['image_file']
            # 生成一个没有乱码的文件名
            secure_filename = werkzeug.secure_filename(img_file.filename)
            # 图片的的保存路径
            img_path = os.path.join(app.root_path, secure_filename)
            # 将图片保存在根目录下
            img_file.save(img_path)
            print('图片上传成功')
            return flask.redirect(flask.url_for(endpoint='predict'))
    except:
        return '上传图片失败, 图像错误'
    return '上传图片失败, 传输错误'


app.add_url_rule(rule = '/upload/', endpoint='upload', view_func = upload_image, methods = ['POST'])

def redirect_upload():
    return flask.render_template(template_name_or_list='upload_image.html')

# 增加默认主页的入口
app.add_url_rule(rule = '/', endpoint='homepage', view_func=redirect_upload)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8808, debug=False)
