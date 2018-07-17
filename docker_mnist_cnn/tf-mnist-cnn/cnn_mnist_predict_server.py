from flask import Flask
from flask import request
from flask import jsonify
from werkzeug import secure_filename

import os
import commands

app = Flask(__name__)



@app.route("/api/v1/predict", methods=["POST"])
def test():
    uploaded_file = request.files["image"]
    uploaded_image = os.path.join("/tmp", secure_filename(uploaded_file.filename))
    uploaded_file.save(uploaded_image)

    (status, output) = commands.getstatusoutput("sh /root/tf-mnist-cnn/cnn_mnist_predict_client " + uploaded_image)
    if status != 0:
        return jsonify(code=500, msg="ERROR", data={"result": output})

    ret = ""
    resultlist = output.split("\n")
    for result in resultlist:
        if result.find("Predict Result") == 0:
            ret = result
            break

    return jsonify(code=0, msg="OK", data={"result": ret})

@app.route("/api/v1/test", methods=["GET"])
def used_ports():
    try:
        testdata = {"name":"testdata"}
        return jsonify(code=0, msg="OK", data=testdata)
    except Exception:
        return jsonify(code=1, msg="InnerError")

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=9000)
