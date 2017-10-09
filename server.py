__author__ = 'tungtt'

from flask import Flask, request
from digit_recognizer import predict_image
from io import open

app = Flask('digit')
@app.route('/',methods = ['GET'])
def homepage():
    return "hello"

@app.route('/digit', methods=['POST'])
def process_request():
    data = request.get_data()
    with open('temp.png','r+b') as f:
        f.write(data)
    x = predict_image()
    return str(x)

if __name__ == '__main__':
    app.run(port=12345)
