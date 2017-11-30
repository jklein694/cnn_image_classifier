import os
import socket

from flask import Flask, request, render_template
from redis import Redis, RedisError

import choices as run

# Connect to Redis
redis = Redis(host="redis", db=0, socket_connect_timeout=2, socket_timeout=2)

app = Flask(__name__)


@app.route('/')
def cnn():
    run.choices()


# submitting the form on home runs this:
@app.route('/runner', methods=['POST', 'GET'])
def runner():
    if request.method == 'POST':
        result = request.form

        try:
            run_what = str(result['run_what'])
        except:
            pass
        # try:
        #     new_vector[index_dict['ORIGIN_' + str(result['origin'])]] = 1
        # except:
        #     pass
        # try:
        #     new_vector[index_dict['DEST_' + str(result['dest'])]] = 1
        # except:
        #     pass
        # try:
        #     new_vector[index_dict['DEP_HOUR_' + str(result['dep_hour'])]] = 1
        # except:
        #     pass

        # pkl_file = open('logmodel.pkl', 'rb')
        # logmodel = pickle.load(pkl_file)
        # prediction = logmodel.predict(new_vector.reshape(1, -1))

        return run_what, render_template('result.html')


def hello():
    try:
        visits = redis.incr("counter")
    except RedisError:
        visits = "<i>cannot connect to Redis, counter disabled</i>"

    html = "<h3>Hello {name}!</h3>" \
           "<b>Hostname:</b> {hostname}<br/>" \
           "<b>Visits:</b> {visits}"
    return html.format(name=os.getenv("NAME", "world"), hostname=socket.gethostname(), visits=visits)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
