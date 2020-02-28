# -*- coding: utf-8 -*-

# -- stdlib --
from logging import StreamHandler
import argparse
import base64
import functools
import logging
import operator
import re
import sys

# -- third party --
from flask import Flask, jsonify, request
from requestlogger import ApacheFormatter, WSGILogger
import bjoern
import msgpack
import numpy as np
import onnxruntime

# -- own --
from check import check_type


# -- code --
log = logging.getLogger('onnxrt-server')
app = Flask('onnxrt-server')
model = None


# https://github.com/microsoft/onnxruntime/blob/v0.4.0/onnxruntime/python/onnxruntime_pybind_mlvalue.cc
ONNX_TO_NP_TYPE = {
    "tensor(float16)": "float16",
    "tensor(float)":   "float32",
    "tensor(double)":  "float64",
    "tensor(int32)":   "int32",
    "tensor(uint32)":  "uint32",
    "tensor(int8)":    "int8",
    "tensor(uint8)":   "uint8",
    "tensor(int16)":   "int16",
    "tensor(uint16)":  "uint16",
    "tensor(int64)":   "int64",
    "tensor(uint64)":  "uint64",
    "tensor(bool)":    "bool",
    "tensor(string)":  "object",
}


class AppError(Exception):

    def __init__(self, message, status_code, payload={}):
        Exception.__init__(self)
        self.message = message
        self.status_code = status_code
        self.payload = {**payload, 'error': message}


@app.errorhandler(AppError)
def handle_app_error(e):
    response = jsonify(e.payload)
    response.status_code = e.status_code
    return response


@app.errorhandler(Exception)
def report_exception(e):
    log.exception('Catched exception')
    return {'error': str(e)}, 500


@app.route('/v1/status')
def model_status():
    return 'not impl'


@app.route('/v1/meta')
def model_metadata():
    inputs = [{
        'name':  i.name,
        'shape': i.shape,
        'onnx_type':  i.type,
        'numpy_type': ONNX_TO_NP_TYPE[i.type],
    } for i in model.get_inputs()]

    outputs = [{
        'name':  i.name,
        'shape': i.shape,
        'onnx_type':  i.type,
        'numpy_type': ONNX_TO_NP_TYPE[i.type],
    } for i in model.get_outputs()]

    meta = model.get_modelmeta()
    meta = {
        'custom_metadata_map': meta.custom_metadata_map,
        'description':         meta.description,
        'domain':              meta.domain,
        'graph_name':          meta.graph_name,
        'producer_name':       meta.producer_name,
        'version':             str(meta.version),
    }

    return {
        'inputs': inputs,
        'outputs': outputs,
        'metadata': meta,
    }


def parse_type(tp):
    if not tp or tp == '*/*':
        return ('base64', 'json')

    m = re.match(r'^application/(?:([a-z0-9]+)\+)?([a-z0-9\.\-]+)(?:[, ;]|$)', tp)
    if not m:
        return None

    fmt, enc = m.groups()

    COALESCE = {
        'msgpack':         'msgpack',
        'x-msgpack':       'msgpack',
        'vnd.msgpack':     'msgpack',
        'vnd.messagepack': 'msgpack',
        'json':            'json',
    }

    if enc not in COALESCE:
        return None

    enc = COALESCE[enc]

    DEFAULT_FMT = {
        'msgpack': 'bytes',
        'json':    'base64',
    }

    fmt = fmt or DEFAULT_FMT.get(enc)

    if fmt not in ('bytes', 'base64', 'plain'):
        return None

    COMPATIBLE  = (
        ('base64', 'json'),
        ('plain',  'json'),

        ('bytes',  'msgpack'),
        ('base64', 'msgpack'),
        ('plain',  'msgpack'),
    )

    if (fmt, enc) not in COMPATIBLE:
        return None

    return (fmt, enc)


def get_payload():
    tp = parse_type(request.mimetype)
    if not tp:
        raise AppError("Invalid Content-Type", 415)

    fmt, enc = tp
    if enc == 'msgpack':
        req = msgpack.unpackb(request.data, raw=False)
    elif enc == 'json':
        req = request.json
    else:
        raise Exception('WTF')

    return fmt, req


def decode_input(model_input, data, fmt):
    assert fmt in ('plain', 'base64', 'bytes')
    i = model_input
    tp = getattr(np, ONNX_TO_NP_TYPE[i.type])

    if fmt == 'plain':
        a = np.asarray(data, dtype=tp)
        if tuple(a.shape) != tuple(i.shape):
            raise AppError(f'Input "{i.name}" shape mismatch: provided({list(a.shape)}) != expected({list(i.shape)})', 400)
        return a

    if fmt == 'base64':
        d = base64.b64decode(data)
    elif fmt == 'bytes':
        if not isinstance(data, bytes):
            raise AppError(f'Input "{i.name}" is not bytes')
        d = data

    a = np.frombuffer(d, dtype=tp)
    expected_size = functools.reduce(operator.mul, i.shape, a.itemsize)
    if a.nbytes != expected_size:
        raise AppError(f'Input "{i.name}" size mismatch: provided({a.nbytes} bytes) != expected({expected_size} bytes)', 400)
    a = a.reshape(*i.shape)
    return a


def make_output(results):
    accept = request.headers['Accept']
    tp = parse_type(accept)
    if not tp:
        raise AppError("Invalid Accept", 406)

    fmt, enc = tp
    rst = {}
    for k, v in results.items():
        if fmt == 'bytes':
            rst[k] = v.tobytes()
        elif fmt == 'base64':
            rst[k] = base64.b64encode(v.tobytes()).decode('utf-8')
        elif fmt == 'plain':
            rst[k] = v.tolist()
        else:
            raise Exception('WTF')

    if enc == 'msgpack':
        payload = msgpack.packb(rst, use_bin_type=True)
    elif enc == 'json':
        payload = rst
    else:
        raise Exception('WTF')

    return payload, 200, {'Content-Type': accept}


@app.route('/v1/predict', methods=['POST'])
def model_predict():
    r'''
    {
        "inputs": {
            "input.1": "AAAA",
            "input.1": "\x00\x00\x00",
            "input.1": [[1,2,3], [4,5,6]],
        },
        "outputs": ["output.1"]
    }
    '''
    fmt, req = get_payload()

    err = check_type({
        'inputs': {
            str: object,
            ...: ...,
        },
        'outputs': [str, ...],
    }, req)

    if err:
        return {'error': f"Bad input: {err}"}, 400

    inputs = req['inputs']
    outputs = req['outputs']
    if not outputs:
        return {'error': f"No output selected"}, 400

    unknown_outputs = set(outputs) - set(i.name for i in model.get_outputs())
    if unknown_outputs:
        return {'error': f"Unknown outputs: {' '.join(unknown_outputs)}"}, 400

    np_inputs = {}
    for i in model.get_inputs():
        if i.name not in inputs:
            return {'error': f"Input '{i.name}' not present"}, 400
        np_inputs[i.name] = decode_input(i, inputs[i.name], fmt)

    pred = model.run(outputs, np_inputs)
    results = dict(zip(outputs, pred))

    return make_output(results)


def load_model(path):
    global model
    model = onnxruntime.InferenceSession(path)


def main():
    parser = argparse.ArgumentParser('onnxrt-server')
    parser.add_argument('--model', default='/models/model.onnx')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8001)
    options = parser.parse_args()

    load_model(options.model)

    wrapped = WSGILogger(app, [StreamHandler(stream=sys.stdout)], ApacheFormatter())

    bjoern.run(wrapped, options.host, options.port)


if __name__ == '__main__':
    main()
