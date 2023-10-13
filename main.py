import tempfile
import os
from flask import Flask, request, redirect, send_file
from skimage import io
import base64
import glob
import numpy as np

app = Flask(__name__)

main_html = """
<html>
<head></head>
<script>
    var mousePressed = false;
    var lastX, lastY;
    var ctx;

    function getRndLetter() {
        var letters = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
        return letters[Math.floor(Math.random() * letters.length)];
    }

    function InitThis() {
        ctx = document.getElementById('myCanvas').getContext("2d");

        numero = getRndLetter();

        document.getElementById('mensaje').innerHTML  = 'Dibujando el numero: ' + numero;
        document.getElementById('numero').value = numero;

        $('#myCanvas').mousedown(function (e) {
            mousePressed = true;
            Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, false);
        });

        $('#myCanvas').mousemove(function (e) {
            if (mousePressed) {
                Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, true);
            }
        });

        $('#myCanvas').mouseup(function (e) {
            mousePressed = false;
        });
            $('#myCanvas').mouseleave(function (e) {
            mousePressed = false;
        });
        
    }

    function Draw(x, y, isDown) {
        if (isDown) {
            ctx.beginPath();
            ctx.strokeStyle = 'red';
            ctx.lineWidth = 10;
            ctx.lineJoin = "round";
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(x, y);
            ctx.closePath();
            ctx.stroke();
        }
        lastX = x; lastY = y;
    }

    function clearArea() {
        // Utilizando la matriz de identidad mientras se limpia el lienzo.
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    }

    function prepareImg() {
        var canvas = document.getElementById('myCanvas');
        document.getElementById('myImage').value = canvas.toDataURL();
    }

    </script>
    <body onload="InitThis();">
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.min.js" type="text/javascript"></script>
        <script type="text/javascript" ></script>
        <div align="left">
        <img src="https://upload.wikimedia.org/wikipedia/commons/f/f7/Uni-logo_transparente_granate.png" width="300"/>
        </div>
        <div align="center">
            <h1 id="mensaje">Dibujando...</h1>
            <canvas id="myCanvas" width="200" height="200" style="border:2px solid black"></canvas>
            <br/>
            <br/>
            <button onclick="javascript:clearArea();return false;">Borrar</button>
        </div>
	<br/>
        <div align="center">
        <form method="post" action="upload" onsubmit="javascript:prepareImg();"  enctype="multipart/form-data">
        <input id="numero" name="numero" type="hidden" value="">
        <input id="myImage" name="myImage" type="hidden" value="">
        <input id="bt_upload" type="submit" value="Enviar!">
        </form>
    </div>
</body>
</html>

"""


@app.route("/")
def main():
    return main_html


@app.route("/upload", methods=["POST"])
def upload():
    try:
        # check if the post request has the file part
        img_data = request.form.get("myImage").replace("data:image/png;base64,", "")
        numero = request.form.get("numero")
        print(numero)
        with tempfile.NamedTemporaryFile(
            delete=False, mode="w+b", suffix=".png", dir=str(numero)
        ) as fh:
            fh.write(base64.b64decode(img_data))
        # file = request.files['myImage']
        print("Image uploaded")
    except Exception as err:
        print("Error occurred")
        print(err)

    return redirect("/", code=302)


@app.route("/prepare", methods=["GET"])
def prepare_dataset():
    images = []
    digitos = []
    folderNames = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    for digito in folderNames:
        filelist = glob.glob("{}/*.png".format(digito))
        images_read = io.concatenate_images(io.imread_collection(filelist))
        images_read = images_read[:, :, :, 3]
        letters_read = np.array([digito] * images_read.shape[0])
        images.append(images_read)
        digitos.append(letters_read)
    images = np.vstack(images)
    digitos = np.concatenate(digitos)
    np.save("X.npy", images)
    np.save("y.npy", digitos)
    return "OK!"


@app.route("/X.npy", methods=["GET"])
def download_X():
    return send_file("./X.npy")


@app.route("/y.npy", methods=["GET"])
def download_y():
    return send_file("./y.npy")


if __name__ == "__main__":
    folderNames = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    for folderName in folderNames:
        if not os.path.exists(str(folderName)):
            os.mkdir(str(folderName))
    app.run()
