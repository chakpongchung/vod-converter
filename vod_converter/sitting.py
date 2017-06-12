"""
Ingestor and egestor for siiting data formats.

"""

import json
import os
import shutil
import xml.etree.ElementTree as ET

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import concurrent
import cv2
import numpy as np
import scipy.io as sio
import scipy
from tqdm import tqdm

from converter import Egestor, Ingestor


class SittingIngestor(Ingestor):

    def loadBboxFromMat(self, filename):
        mat = scipy.io.loadmat(filename, struct_as_record=True)
        img = mat['M']
        # print img[np.nonzero(img)]
        img = img.astype(np.uint8)
        # for i in range(img.shape[0]):
        # 	for j in range(img.shape[1]):
        # 		if img[i][j]>0:
        # 			print img[i][j]
        # cv2.imshow('maat',img)
        # cv2.waitKey(0)

        # cv2.imwrite('test.jpg',img)
        ret, thresh = cv2.threshold(img, 0, 255, 0)

        im2, contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print len(contours)

        imdis = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # c=max(contours, key = cv2.contourArea)

        area = []
        print (len(contours))
        for i in range(len(contours)):
            area.append(cv2.contourArea(contours[i]))

        # print (np.argmax(area))
        cnt = contours[np.argmax(area)]
        # cv2.drawContours(imdis, [cnt], 0, (0,255,0), 1)
        # cv2.drawContours(imdis, contours, -1, (0,255,0), 3)

        # cv2.imshow('maat',imdis)
        # cv2.waitKey(0)

        x, y, w, h = cv2.boundingRect(cnt)
        img = cv2.rectangle(imdis, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # cv2.imshow('maat',img)
        # cv2.waitKey(0)

        return x, y, w, h

    def loading(self, name):
        # name = os.path.splitext(
        #     self.mat_contents['RAP_annotation']['imagesname'][0][0][i][0][0])[0]
        
        # positions = self.mat_contents['RAP_annotation']['position'][0][0][i]


        img = cv2.imread(self.RAPpath + "/data/FaceDetectData/CMStest/sitting/img/" + name + ".jpg")
        # print (self.RAPpath + "/data/FaceDetectData/CMStest/sitting/img/" + name + ".jpg")
        height, width = img.shape[:2]

        self.d[name] = {}
        self.d[name]['height'] = height
        self.d[name]['width'] = width
        self.d[name]['positions'] = self.mat_contents[name]
        # self.d[name]['positions'] = self.mat_contents[name].tolist()

        if self.d[name]['positions'][2] > width:
            self.d[name]['positions'][2] = width

        if self.d[name]['positions'][3] > height:
            self.d[name]['positions'][3] = height

    def __init__(self):
        print("__init__ from sitting began loading .mat into memory!")

        self.RAPpath = "/local/home/cpchung"

        self.mat_contents = {}

        # self.mat_contents = sio.loadmat(
        #     self.RAPpath + '/RAP_annotation/RAP_annotation.mat')

        rootDir = self.RAPpath + '/data/FaceDetectData/CMStest/sitting/masks/'
        for dirName, subdirList, fileList in os.walk(rootDir):
            # print('Found directory: %s' % dirName)
            if subdirList != []:
                continue

            for fname in fileList:
                if fname.endswith(('.mat')):
                    x, y, w, h = self.loadBboxFromMat(rootDir + fname)
                    # print(fname, x, y, w, h)

                    self.mat_contents[fname.replace(".mat",'')] = [x, y, w, h]

        # print (self.mat_contents)


        numOfImgs = len(self.mat_contents)
        # input("Press Enter to continue...")

        self.d = {}
        DEBUG = 0
        if not DEBUG:
            # for i in tqdm(range(numOfImgs)):
            #     self.loading(i)
            print (multiprocessing.cpu_count())
            with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count() * 2 / 3) as executor:
                # future_to_url = {executor.submit(
                #     self.loading, i): i for i in tqdm(range(numOfImgs))}

                future_to_url = {executor.submit(
                    self.loading, name): name for name in self.mat_contents}

                

                kwargs = {
                    'total': len(future_to_url),
                    'unit': 'it',
                    'unit_scale': True,
                    'leave': True
                }
                for future in tqdm(concurrent.futures.as_completed(future_to_url), **kwargs):
                    url = future_to_url[future]
                    try:
                        data = future.result()
                    except Exception as exc:
                        print('%r generated an exception: %s' % (url, exc))
                    else:
                        # print('%r page is %d bytes' % (url, len(data)))
                        pass

            # for i in tqdm(range(numOfImgs)):
                # name = os.path.splitext(
                #     mat_contents['RAP_annotation']['imagesname'][0][0][i][0][0])[0]
                # positions = mat_contents['RAP_annotation']['position'][0][0][i]
                # img = cv2.imread(RAPpath + "/RAP_dataset/" + name + ".png")
                # height, width = img.shape[:2]

                # self.d[name] = {}
                # self.d[name]['height'] = height
                # self.d[name]['width'] = width
                # self.d[name]['positions'] = positions.tolist()

                # if self.d[name]['positions'][2] > width:
                #     self.d[name]['positions'][2] = width

                # if self.d[name]['positions'][3] > height:
                #     self.d[name]['positions'][3] = height

        #  FOR DEBUG ONLY
        if DEBUG:
            name = 'CAM31_2014-03-18_20140318125002-20140318125546_tarid91_frame2631_line1'
            self.d[name] = {}
            self.d[name]['width'] = 116
            self.d[name]['height'] = 288
            self.d[name]['positions'] =\
                [583, 5, 116, 288,
                 590, 6, 97, 71,
                 584, 51, 114, 138,
                 604, 137, 92, 152]

            name = 'CAM31_2014-03-18_20140318125002-20140318125546_tarid93_frame2823_line1'
            self.d[name] = {}
            self.d[name]['width'] = 124
            self.d[name]['height'] = 254
            self.d[name]['positions'] = \
                [693, 16, 124, 254,
                 731, 21, 54, 49,
                 708, 56, 102, 113,
                 726, 149, 62, 116]

            name = 'CAM29_2014-02-22_20140222103914-20140222104502_tarid608_frame4080_line2'
            self.d[name] = {}
            self.d[name]['width'] = 221
            self.d[name]['height'] = 361
            self.d[name]['positions'] =\
                [933, 358, 221, 361,
                 1025, 360, 123, 163,
                 935, 456, 210, 263,
                 935, 718, 4, 1]

            # 2,98,212,361

        print("__init__ finished loading .mat into memory!")

    def validate(self, root):
        # path = f"{root}/VOC2012"
        return True, None

    def ingest(self, path):

        print ('begin to ingest()')
        image_names = self._get_image_ids(path)

        # res=list(filter(None, (self._get_image_detection(path, image_name) for image_name in tqdm(image_names))))
        # print ('list of images',len(res))

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:

            future_to_url = {executor.submit(
                self._get_image_detection, path, image_name): image_name for image_name in tqdm(image_names)}
            res = []

            kwargs = {
                'total': len(image_names),
                'unit': 'it',
                'unit_scale': True,
                'leave': True
            }
            for future in tqdm(concurrent.futures.as_completed(future_to_url), **kwargs):
                url = future_to_url[future]
                try:
                    data = future.result()
                    # print ('self._get_image_detection',data)
                    if data is not None:
                        res.append(data)
                except Exception as exc:
                    print('%r generated an exception: %s' % (url, exc))
                else:
                    # print('%r page is %d bytes' % (url, len(data)))
                    pass
        print ('list of images', len(res))
        return res

    def _get_image_ids(self, root):
        # print (len(list(self.d.keys())))
        return list(self.d.keys())

    def _get_image_detection(self, root, image_id):

        path = root
        # image_path = f"{path}/RAP_dataset/{image_id}.jpg"
        image_path = f"{path}/img/{image_id}.jpg"
        
        if not os.path.isfile(image_path):
            raise Exception(f"Expected {image_path} to exist.")

        segmented_path = None

        image_width = self.d[image_id]['width']
        image_height = self.d[image_id]['height']

        if (image_width < 100 or image_height < 55):
            return

        a = self.d[image_id]['positions']
        n = 4

        g4 = [a[k:k + n] for k in range(0, len(a), n)]

        # print (input("Press Enter to continue..."))
        bbox = list(filter(None, (self._get_detection(i, g4, image_path)
                                  for i in range(len(g4)))))

        if bbox == []:
            print('empty bbox', image_path)
            return

        return {
            'image': {
                'id': image_id,
                'path': image_path,
                'segmented_path': segmented_path,
                'width': image_width,
                'height': image_height
            },
            'detections': bbox
            # list(filter(func,data)) #python 3.x

        }

    def _get_detection(self, i, g4, image_path):

            # bounding box not visible
        if all(v == 0 for v in g4[i]) or i == 8880:
            # print ('invisible: ',image_path)
            return

            # bounding box is a thin slice
        if g4[i][2] <= 3 or g4[i][3] <= 3:
            # print ('thin slice: ',image_path)
            return

        # handling bbox margins
        margin = 0

        if g4[i][0] - g4[0][0] < int(margin * g4[0][2]):
            print('returned','image_path')
            
            return
        else:
            xmin = g4[i][0] - g4[0][0]

        if g4[i][1] - g4[0][1] < int(margin * g4[0][3]):
            return
        else:
            ymin = g4[i][1] - g4[0][1]

        if g4[i][0] - g4[0][0] + g4[i][2] > int((1 - margin) * g4[0][2]):
            return
        else:
            xmax = g4[i][0] - g4[0][0] + g4[i][2] - 1

        if g4[i][1] - g4[0][1] + g4[i][3] > int((1 - margin) * g4[0][3]):
            return
        else:
            ymax = g4[i][1] - g4[0][1] + g4[i][3] - 1

        if (xmax - xmin) < 10 or (ymax - ymin) < 10:
            return

        label = ['fullBody', 'headShoulder', 'upperBoddy', 'lowerBody']

        # img = cv2.imread(image_path, 1)
        # height, width = img.shape[:2]

        # scale=3
        # extended =(scale-1)/scale/2

        # xmin=xmin+int(extended*width)
        # ymin=ymin+int(extended*height)
        # xmax=xmax+int(extended*width)
        # ymax=ymax+int(extended*height)

        # if not all(i >= 100 for i in [xmin,ymin,xmax,ymax]):
        #     print ('coord <100: ',image_path)

        #     cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,255,0),2)

        #     # cv2.imshow('image',img)
        #     # cv2.waitKey()
        #     cv2.imwrite('./temp/'+image_path.split('/')[-1],img)
        #     return

        ret = {
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


def add_sub_node(node, name, kvs):
    subnode = ET.SubElement(node, name)
    for k, v in kvs.items():
        add_text_node(subnode, k, v)
    return subnode


def add_text_node(node, name, text):
    subnode = ET.SubElement(node, name)
    subnode.text = f"{text}"
    return subnode
