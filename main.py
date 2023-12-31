import base64
import glob
import os
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, send_file
from skimage import io

app = Flask(__name__)

main_html = """
<html>
<head></head>
<script>
    var mousePressed = false;
    var lastX, lastY;
    var ctx;

    function getRndLetter() {
        var letters = ["1","2", "3"];
        return letters[Math.floor(Math.random() * letters.length)];
    }

    function InitThis() {
        ctx = document.getElementById('myCanvas').getContext("2d");

        letra = getRndLetter();

        document.getElementById('mensaje').innerHTML  = 'Dibuje un numero del 0 al 9';
        document.getElementById('letra').value = letra;

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
            ctx.strokeStyle = 'black';
            ctx.lineWidth = 11;
            ctx.lineJoin = "round";
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(x, y);
            ctx.closePath();
            ctx.stroke();
        }
        lastX = x; lastY = y;
    }

    function clearArea() {
        // Use the identity matrix while clearing the canvas
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    }

    //https://www.askingbox.com/tutorial/send-html5-canvas-as-image-to-server
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
        <div align="center">
        <form method="post" action="predict" onsubmit="javascript:prepareImg();"  enctype="multipart/form-data">
        <input id="letra" name="letra" type="hidden" value="">
        <input id="myImage" name="myImage" type="hidden" value="">
        <input id="bt_upload" type="submit" value="Predecir">
        </form>
    </div>
</body>
</html>

"""

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

@app.route("/")
def main():
    print("aaaaaaaaaaaaaaaaaaaaaaaaa")
    return main_html

@app.route("/predict", methods=["POST"])
def upload():
    prediccion=-1
    try:
        print("aaaaaaaaaaaaaaaaaaaaaaaaa")
        # check if the post request has the file part
        img_data = request.form.get("myImage").replace("data:image/png;base64,", "")
        img_bytes = base64.b64decode(img_data)
        image = Image.open(BytesIO(img_bytes))

        # Redimensiona la imagen a 28x28 píxeles
        image = image.resize((28, 28), Image.ANTIALIAS)
        # Convierte la imagen a un array de NumPy y asegura que tenga una sola dimensión de color
        image_array = np.array(image)

        plt.imshow(image_array, cmap='gray')
        plt.show()

        # Asegura que la imagen sea de forma (28, 28, 1) si originalmente tenía una forma de (200, 200, 4)
        if image_array.shape[-1] == 4:
            # Si la imagen tiene 4 canales (RGBA), mantén solo el canal de escala de grises (primer canal)
            # print(image_array[:, :, 3])
            # plt.imshow(image_array[:, :, 3])
            # plt.show()
            image_array = image_array[:, :, 3] / 255
            image_array = image_array[:, :, None]
            print(image_array)


        plt.imshow(image_array, cmap='gray')
        plt.show()

        model = tf.keras.models.load_model('mi_modelo.h5')

        # Realizar predicciones
        salida = model.predict(image_array[None, :, :, :])[0]
        print(salida.argmax())
        prediccion=salida.argmax()

    except Exception as err:
        print("Error occurred")
        print(err)

    return f"""
        <html>
        <head></head>
        <body>
            <p>{prediccion}</p>
        </bo1dy>
        </html>
    """


@app.route("/prepare", methods=["GET"])
def prepare_dataset():
    images = []
    letters = []
    folderNames = ["1", "2", "3"]
    for letter in folderNames:
        filelist = glob.glob("{}/*.png".format(letter))
        images_read = io.concatenate_images(io.imread_collection(filelist))
        images_read = images_read[:, :, :, 3]
        letters_read = np.array([letter] * images_read.shape[0])
        images.append(images_read)
        letters.append(letters_read)
    images = np.vstack(images)
    letters = np.concatenate(letters)
    np.save("X.npy", images)
    np.save("y.npy", letters)
    return "OK!"


@app.route("/X.npy", methods=["GET"])
def download_X():
    return send_file("./X.npy")


@app.route("/y.npy", methods=["GET"])
def download_y():
    return send_file("./y.npy")


if __name__ == "__main__":
    folderNames = ["1", "2", "3"]
    for folderName in folderNames:
        if not os.path.exists(str(folderName)):
            os.mkdir(str(folderName))
    # app.run(debug=False, host="0.0.0.0", port=8000)
    app.run()
