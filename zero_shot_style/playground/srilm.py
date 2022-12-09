import os
senticap_dataset = os.path.join(os.path.expanduser('~'), 'zero-shot-style', 'data', 'senticap_dataset','data','senticap_dataset.json')
import json

with open(senticap_dataset) as f:
    data = json.load(f)
print('finish')