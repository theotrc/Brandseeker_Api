from App import app, util
from flask import request

@app.route('/predict', methods=['get', 'post'])
def prediction():
    record = request.json
    util.pytube_dl(str(record))
    util.prediction_pdf()
    return "ok"