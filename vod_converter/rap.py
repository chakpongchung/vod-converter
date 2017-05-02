"""
Ingestor and egestor for RAP formats.

http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/index.html
"""

import os
import shutil

from converter import Ingestor, Egestor
import xml.etree.ElementTree as ET
import cv2
import json
import scipy.io as sio
import os
import json
import numpy as np
from tqdm import tqdm


class RAPIngestor(Ingestor):
    

    def __init__(self):
        print("__init__ began loading .mat into memory!")

        RAPpath="/local/home/cpchung/data/pedestrian"
        mat_contents = sio.loadmat(RAPpath+'/RAP_annotation/RAP_annotation.mat')
        numOfImgs= len(mat_contents['RAP_annotation']['imagesname'][0][0])
        self.d={}
        DEBUG=0
        if not DEBUG:
            for i in tqdm(range(numOfImgs)):
                name=os.path.splitext(mat_contents['RAP_annotation']['imagesname'][0][0][i][0][0])[0]
                positions=mat_contents['RAP_annotation']['position'][0][0][i]
                img=cv2.imread(RAPpath+"/RAP_dataset/"+name+".png")
                height, width = img.shape[:2]

                self.d[name]={}
                self.d[name]['height']=height
                self.d[name]['width']=width
                self.d[name]['positions']=positions.tolist()

                if self.d[name]['positions'][2]>width:
                    self.d[name]['positions'][2]=  width

                if self.d[name]['positions'][3]>height:
                    self.d[name]['positions'][3]=  height            
        
        #  FOR DEBUG ONLY         
        if DEBUG:    
            name='CAM31_2014-03-18_20140318125002-20140318125546_tarid91_frame2631_line1'
            self.d[name]={}
            self.d[name]['width']=116
            self.d[name]['height']=288
            self.d[name]['positions']=\
            [583,5,116,288, \
            590,6,97,71,\
            584,51, 114,138,\
            604,137,92,152]


            name='CAM31_2014-03-18_20140318125002-20140318125546_tarid93_frame2823_line1'
            self.d[name]={}
            self.d[name]['width']=124
            self.d[name]['height']=254
            self.d[name]['positions']= \
            [693, 16, 124, 254, \
            731, 21, 54, 49, \
            708, 56, 102, 113,\
            726, 149, 62, 116]

            name='CAM29_2014-02-22_20140222103914-20140222104502_tarid608_frame4080_line2'
            self.d[name]={}
            self.d[name]['width']=221
            self.d[name]['height']=361
            self.d[name]['positions']=\
            [933, 358, 221, 361,\
             1025, 360, 123, 163,\
             935, 456, 210, 263, \
             935, 718, 4, 1] 

             #2,98,212,361



            
        print("__init__ finished loading .mat into memory!")


    def validate(self, root):
        # path = f"{root}/VOC2012"
        return True, None

    def ingest(self, path):
        image_names = self._get_image_ids(path)
        # print (image_names,"image_names")
        return [self._get_image_detection(path, image_name) for image_name in image_names]

    def _get_image_ids(self, root):
        # print (len(list(self.d.keys())))
        return list(self.d.keys())

    def _get_image_detection(self, root, image_id):
        path=root
        image_path = f"{path}/RAP_dataset/{image_id}.jpg"
        if not os.path.isfile(image_path):
            raise Exception(f"Expected {image_path} to exist.")

        # annotation_path = f"{path}/RAP_annotation/{image_id}.xml"

        # if not os.path.isfile(annotation_path):
        #     raise Exception(f"Expected annotation file {annotation_path} to exist.")
        # tree = ET.parse(annotation_path)
        # xml_root = tree.getroot()
        # size = xml_root.find('size')
        # segmented = xml_root.find('segmented').text == '1'
        segmented_path = None
        # if segmented:
        #     segmented_path = f"{path}/SegmentationObject/{image_id}.png"
        #     if not os.path.isfile(segmented_path):
        #         raise Exception(f"Expected segmentation file {segmented_path} to exist.")
        # image_width = int(size.find('width').text)
        # image_height = int(size.find('height').text)

        # img = cv2.imread(image_path,0)
	    # image_height, image_width = img.shape[:2]
        
        image_width =self.d[image_id]['width']
        image_height=self.d[image_id]['height']
        a = self.d[image_id]['positions']
        n = 4
        g4 = [a[k:k+n] for k in range(0,len(a),n)]

        # for i in range(len(g4)):
        #     print (self._get_detection(i,g4) )
        # print (list(filter(None, (self._get_detection(i,g4) for i in range(len(g4))))))
        # print (input("Press Enter to continue..."))

        return {
            'image': {
                'id': image_id,
                'path': image_path,
                'segmented_path': segmented_path,
                'width': image_width,
                'height': image_height
            },
            # 'detections': [self._get_detection(i,g4) for i in range(len(g4))]
            'detections': list(filter(None, (self._get_detection(i,g4) for i in range(len(g4)))))
            # list(filter(func,data)) #python 3.x

        }

    def _get_detection(self, i , g4):
        if all(v == 0 for v in g4[i]) or i==0:
            # bounding box not visible
            return


        if g4[i][2]<=3 or g4[i][3]<=3 :
            # bounding box is a thin slice
            return
        xmin= max(g4[i][0]-g4[0][0],int(0.35*g4[0][2]))
        ymin= max(g4[i][1]-g4[0][1],int(0.35*g4[0][3]))
        xmax= min(g4[i][0]-g4[0][0]+g4[i][2]-1,int(0.65*g4[0][2])-1)
        ymax= min(g4[i][1]-g4[0][1]+g4[i][3]-1,int(0.65*g4[0][3])-1)

        if (xmax-xmin)<10 or (ymax-ymin)<10:
            return
        label=['fullBody','headShoulder','upperBoddy','lowerBody']
        ret= {
            # 'label': node.find('name').text,
            # 'left': float(bndbox.find('xmin').text) - 1,
            # 'top': float(bndbox.find('ymin').text) - 1,
            # 'right': float(bndbox.find('xmax').text) - 1,
            # 'bottom': float(bndbox.find('ymax').text) - 1,
            'label': label[i],
            'left':   xmin,
            'top':    ymin,
            'right':  xmax,
            'bottom': ymax,
            }
        return ret


class RAPEgestor(Egestor):

    def expected_labels(self):
        return {
            'aeroplane': [],
            'bicycle': [],
            'bird': [],
            'boat': [],
            'bottle': [],
            'bus': [],
            'car': [],
            'cat': [],
            'chair': [],
            'cow': [],
            'diningtable': [],
            'dog': [],
            'horse': [],
            'motorbike': [],
            'person': ['pedestrian'],
            'pottedplant': [],
            'sheep': [],
            'sofa': [],
            'train': [],
            'tvmonitor': []
        }

    def egest(self, *, image_detections, root):
        image_sets_path = f"{root}/VOC2012/ImageSets/Main"
        images_path = f"{root}/VOC2012/JPEGImages"
        annotations_path = f"{root}/VOC2012/Annotations"
        segmentations_path = f"{root}/VOC2012/SegmentationObject"
        segmentations_dir_created = False

        for to_create in [image_sets_path, images_path, annotations_path]:
            os.makedirs(to_create, exist_ok=True)

        for image_detection in image_detections:
            image = image_detection['image']
            image_id = image['id']
            src_extension = image['path'].split('.')[-1]
            shutil.copyfile(image['path'], f"{images_path}/{image_id}.{src_extension}")

            with open(f"{image_sets_path}/trainval.txt", 'a') as out_image_index_file:
                out_image_index_file.write(f'{image_id}\n')

            if image['segmented_path'] is not None:
                if not segmentations_dir_created:
                    os.makedirs(segmentations_path)
                    segmentations_dir_created = True
                shutil.copyfile(image['segmented_path'], f"{segmentations_path}/{image_id}.png")

            xml_root = ET.Element('annotation')
            add_text_node(xml_root, 'filename', f"{image_id}.{src_extension}")
            add_text_node(xml_root, 'folder', 'VOC2012')
            add_text_node(xml_root, 'segmented', int(segmentations_dir_created))

            add_sub_node(xml_root, 'size', {
                'depth': 3,
                'width': image['width'],
                'height': image['height']
            })
            add_sub_node(xml_root, 'source', {
                'annotation': 'Dummy',
                'database': 'Dummy',
                'image': 'Dummy'
            })

            for detection in image_detection['detections']:
                x_object = add_sub_node(xml_root, 'object', {
                    'name': detection['label'],
                    'difficult': 0,
                    'occluded': 0,
                    'truncated': 0,
                    'pose': 'Unspecified'
                })
                add_sub_node(x_object, 'bndbox', {
                    'xmin': detection['left'] + 1,
                    'xmax': detection['right'] + 1,
                    'ymin': detection['top'] + 1,
                    'ymax': detection['bottom'] + 1
                })

            ET.ElementTree(xml_root).write(f"{annotations_path}/{image_id}.xml")


def add_sub_node(node, name, kvs):
    subnode = ET.SubElement(node, name)
    for k, v in kvs.items():
        add_text_node(subnode, k, v)
    return subnode


def add_text_node(node, name, text):
    subnode = ET.SubElement(node, name)
    subnode.text = f"{text}"
    return subnode







