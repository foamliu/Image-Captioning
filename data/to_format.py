# -*- coding: utf-8 -*-
import json
import pickle

with open('preds.p', 'rb') as file:
    preds = pickle.load(file)

print(len(preds))

result = []
for pred in preds:
    candidate = pred['candidate']
    image_name = pred['image_name']
    caption = ''.join(candidate.split())
    image_id = image_name.split('.')[0]
    result.append({'caption': caption, 'image_id': image_id})

with open('submit_data.json', 'w') as file:
    json.dump(result, file, indent=4, ensure_ascii=False)
