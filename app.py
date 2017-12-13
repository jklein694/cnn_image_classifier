import os

from flask import Flask, request, render_template
from redis import Redis, RedisError
import time


app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
print(UPLOAD_FOLDER)
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

import check

model_path = "conv_model/model.ckpt"

# Connect to Redis
redis = Redis(host="redis", db=0, socket_connect_timeout=2, socket_timeout=2)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index_landing.html')
    # letting flask know where to find the home template


@app.route('/forms.html')
def form():
    return render_template('forms.html')


@app.route('/portfolio_single_featured_image2.html')
def results():
    return render_template('portfolio_single_featured_image2.html')

    # letting flask know where to find the home template


@app.route('/runner', methods=['POST', 'GET'])
def runner():
    file = request.files['image']
    print(file.filename)
    file.filename = 'name.jpg'
    f = file.save(os.path.join(UPLOAD_FOLDER, file.filename))


    file.save(f)
    # add your custom code to check that the uploaded file is a valid image and not a malicious file (out-of-scope for this post)

    print(file.filename)

    try:
        visits = redis.incr("counter")
    except RedisError:
        visits = "<i>cannot connect to Redis, counter disabled</i>"


    image_size = 64

    if request.method == 'POST':
        result = request.form

        try:
            run_what = result['run_what']
            print(run_what)
        except:
            pass



        file_path = str(f)

        pred, out_text = check.run(UPLOAD_FOLDER, model_path, int(image_size))

        path = 'uploads/' + str(os.listdir(UPLOAD_FOLDER)[0])


        if pred:
            correct_sum = "Great Success! My Neural Network worked. I developed this program in TensorFlow on my personal computer." \
                          " with limited computational power. I would like to run my program on a cloud server, to increase " \
                          "the accuracy. A more powerful computer would allow for more interconnected layers and larger image " \
                          "sizes."
            return render_template('portfolio_single_featured_image2.html', lola=out_text, summary=correct_sum,
                                   file_path=path)

        else:
            wrong_sum = ' Bummer... My Neural Network did not work. I developed this program in TensorFlow on my personal' \
                        ' computer.' \
                        ' with limited computational power. I would like to run my program on a cloud server, to increase' \
                        'the accuracy. A more powerful computer would allow for more interconnected layers and larger image' \
                        'sizes.'
            return render_template('portfolio_single_featured_image2.html', lola=out_text, summary=wrong_sum,
                                   file_path=path)



# @app.route('/')
# def cnn():
#     run.choices()
#
#
# def hello():
#     try:
#         visits = redis.incr("counter")
#     except RedisError:
#         visits = "<i>cannot connect to Redis, counter disabled</i>"
#
#     html = "<h3>Hello {name}!</h3>" \http://0.0.0.0:80/
#            "<b>Hostname:</b> {hostname}<br/>" \
#            "<b>Visits:</b> {visits}"
#     return html.format(name=os.getenv("NAME", "world"), hostname=socket.gethostname(), visits=visits)


if __name__ == "__main__":
    app.debug = True
    app.run()
