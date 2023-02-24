from pathlib import Path
import requests
import zipfile
from pycocotools.coco import COCO
Path("data/annotations/annotations_trainval2014").mkdir(parents=True, exist_ok=True)
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/final/images").mkdir(parents=True, exist_ok=True)

to_download = False

if to_download == True:
    r = requests.get("http://images.cocodataset.org/annotations/annotations_trainval2014.zip")
    open("data/raw/annotations_trainval2014.zip", "wb").write(r.content)

    with zipfile.ZipFile("data/raw/annotations_trainval2014.zip", 'r') as zip_ref:
        zip_ref.extractall("data/annotations/annotations_trainval2014")

coco_ann= COCO('data/annotations/annotations_trainval2014/annotations/instances_train2014.json')
coco_cap = COCO('data/annotations/annotations_trainval2014/annotations/captions_train2014.json')

# Specify a list of category names of interest
category_names = ['person', 'car', 'truck', 'traffic light', 'dog', 'horse', 'zebra','bench','elephant','handbag', 'umbrella']

import json
thresholds = [35, 15, 25, 5, 25, 25, 25, 15, 15, 35, 15]

final_dict = []
img_mapping = {}
for cat_name, threshold in zip(category_names, thresholds):
    catIds = coco_ann.getCatIds(catNms=[cat_name])
    cat_id = catIds[0]
    imgIds = coco_ann.getImgIds(catIds=cat_id)
    img_mapping[cat_id]=imgIds
    
for cat_name, threshold in zip(category_names, thresholds):
    # Get the corresponding image ids and images using loadImgs
    no_of_images = 1000
    count_saved = 0
    count_discarded = 0
    catIds = coco_ann.getCatIds(catNms=[cat_name])
    cat_id = catIds[0]
    imgIds = img_mapping[cat_id]
    images = coco_ann.loadImgs(imgIds)
    supercategory = coco_ann.loadCats([cat_id])[0]['supercategory']
    Path(f"data/final/images/{cat_name}").mkdir(parents=True, exist_ok=True)
    count = 0
    for image in images:
        not_found_in_other = True
        for k,v in img_mapping.items():
            if k !=cat_id:
                if image['id'] in v:
                    not_found_in_other = False
                    break
        if not_found_in_other==False:
            count_discarded+=1
            continue
        image_dict = {}
        ann_ids = coco_cap.getAnnIds(image['id'])
        cap_list = coco_cap.loadAnns(ann_ids)
        ann_ids = coco_ann.getAnnIds(imgIds=[image['id']], iscrowd=None)
        anns = coco_ann.loadAnns(ann_ids)
        main_category_share = sum([a['bbox'][2]*a['bbox'][3]/(image['height']*image['width'])  for a in anns if a['category_id']==cat_id])*100
        if main_category_share>=threshold:
            count_saved+=1
        else:
            count_discarded+=1
            continue
        img_data = requests.get(image['coco_url']).content
        with open(f'data/final/images/{cat_name}/' + image['file_name'], 'wb') as handler:
            handler.write(img_data)
        image_dict['id']=image['id']
        image_dict['category_id']=cat_id
        image_dict['category_name']=cat_name
        image_dict['file']=image['file_name']
        image_dict['coco_url'] = image['coco_url']
        image_dict['height'] = image['height']
        image_dict['width'] = image['width']
        image_dict['supercategory'] = supercategory
        for i in range(len(cap_list)):
            image_dict[f'caption_{i}'] = cap_list[i]['caption']
        print(image_dict["coco_url"], image_dict["category_name"])
        final_dict.append(image_dict)
        count+=1
        if count==no_of_images:
            break
    print(f"Images saved: {count_saved}, Images discarded: {count_discarded}")
with open('data/final/final_dict.json', 'w') as fp:
    json.dump(final_dict, fp)

with open('data/final/final_dict.json') as json_file:
    data = json.load(json_file)

import pandas as pd
data_df = pd.DataFrame(data)
print(data_df.head())