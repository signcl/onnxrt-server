import sys
assert sys.version_info.major == 3

import requests

data = [[[0.2]*5]*4]*3

# -----
import json
resp = requests.post(
    'http://localhost:8001/v1/predict',
    headers={
        'Accept': 'application/plain+json',
        'Content-Type': 'application/plain+json',
    },
    data=json.dumps({'inputs': {'x': data}, 'outputs': ['y']}),
)
print(resp.json())

# -----
import base64
import numpy as np
a = np.empty((3, 4, 5), dtype=np.float32)
a[:] = 0.2

resp = requests.post(
    'http://localhost:8001/v1/predict',
    headers={
        'Accept': 'application/plain+json',
        'Content-Type': 'application/base64+json',
    },
    data=json.dumps({
        'inputs': {'x': base64.b64encode(a.tobytes()).decode('utf-8')},
        'outputs': ['y']
    }),
)
print(resp.json())

# -----
import msgpack
resp = requests.post(
    'http://localhost:8001/v1/predict',
    headers={
        'Accept': 'application/plain+json',
        'Content-Type': 'application/bytes+msgpack',
    },
    data=msgpack.packb({
        'inputs': {'x': a.tobytes()},
        'outputs': ['y'],
    }, use_bin_type=True)
)
print(resp.json())

# -----
import msgpack
import io

npydata = io.BytesIO()
np.save(npydata, a)

resp = requests.post(
    'http://localhost:8001/v1/predict',
    headers={
        'Accept': 'application/plain+json',
        'Content-Type': 'application/npy+msgpack',
    },
    data=msgpack.packb({
        'inputs': {'x': npydata.getvalue()},
        'outputs': ['y'],
    }, use_bin_type=True)
)
print(resp.json())
