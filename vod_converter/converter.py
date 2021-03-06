"""
Defines the protocol for converting too and from a common data format and executes
the conversion, validating proper conversion along the way.

For a given dataformat, e.g `voc.py`, if you wish to support reading in of your data format, define
an `Ingestor` that can read in data from a path and return an array of data conforming to `IMAGE_DETECTION_SCHEMA`.

If you wish to support data output, define an `Egestor` that, given an array of data of the same form,
can output the data to the filesystem.

See `main.py` for the supported types, and `voc.py` and `kitti.py` for reference.
"""
from jsonschema import validate as raw_validate
from jsonschema.exceptions import ValidationError as SchemaError
from tqdm import tqdm
import concurrent

import cv2


def validate_schema(data, schema):
    """Wraps default implementation but accepting tuples as arrays too.

    https://github.com/Julian/jsonschema/issues/148
    """
    return raw_validate(data, schema, types={"array": (list, tuple)})


IMAGE_SCHEMA = {
    'type': 'object',
    'properties': {
        'id': {'type': 'string'},
        'path': {'type': 'string'},
        'segmented_path': {
            'anyOf': [
                {'type': 'null'},
                {'type': 'string'}
            ]
        },
        'width': {'type': 'integer', 'minimum': 10},
        'height': {'type': 'integer', 'minimum': 10},
    },
    'required': ['id', 'path', 'segmented_path', 'width', 'height']
}


DETECTION_SCHEMA = {
    'type': 'object',
    'properties': {
        'label': {'type': 'string'},
        'top': {'type': 'number', 'minimum': 0},
        'left': {'type': 'number', 'minimum': 0},
        'right': {'type': 'number', 'minimum': 0},
        'bottom': {'type': 'number', 'minimum': 0}
    },
    'required': ['top', 'left', 'right', 'bottom']
}

IMAGE_DETECTION_SCHEMA = {
    'type': 'object',
    'properties': {
        'image': IMAGE_SCHEMA,
        'detections': {
            'type': 'array',
            'items': DETECTION_SCHEMA
        }
    }
}


class Ingestor:
    def validate(self, path):
        """
        Validate that a path contains files / directories expected for a given data format.

        This is where you can provide feedback to the end user if they are attempting to convert from
        your format but have passed you path to a directory that is missing the expected files or directory
        structure.

        :param path: Where the data is stored
        :return: (sucess, error message), e.g (False, "error message") if anything is awry, (True, None) otherwise.
        """
        return True, None

    def ingest(self, path):
        """
        Read in data from the filesytem.
        :param path: '/path/to/data/'
        :return: an array of dicts conforming to `IMAGE_DETECTION_SCHEMA`
        """
        pass


class Egestor:

    def expected_labels(self):
        """
        Return a dict with a key for each label generally expected by this dataset format and
        any aliases that should be converted.

        In the example below the expected labels are 'car' and 'pedestrian' and, for example, both
        'Car' and 'auto' should be converted to 'car'.

        :return: {'car': ['Car', 'auto'], 'pedestrian': ['Person']}
        """
        raise NotImplementedError()

    def egest(self, *, image_detections, root):
        """
        Output data to the filesystem.

        Note: image_detections will already have any conversions specified via `expected_labels` applied
        by the time they are passed to this method.

        :param image_detections: an array of dicts conforming to `IMAGE_DETECTION_SCHEMA`
        :param root: '/path/to/output/data/'
        """
        raise NotImplementedError()


def convert(*, from_path, ingestor, to_path, egestor, select_only_known_labels, filter_images_without_labels):
    """
    Converts between data formats, validating that the converted data matches
    `IMAGE_DETECTION_SCHEMA` along the way.

    :param from_path: '/path/to/read/from'
    :param ingestor: `Ingestor` to read in data
    :param to_path: '/path/to/write/to'
    :param egestor: `Egestor` to write out data
    :return: (success, message)
    """
    from_valid, from_msg = ingestor.validate(from_path)

    if not from_valid:
        return from_valid, from_msg

    image_detections = ingestor.ingest(from_path)
    validate_image_detections(image_detections)
    image_detections = convert_labels(
        image_detections=image_detections, expected_labels=egestor.expected_labels(),
        select_only_known_labels=select_only_known_labels,
        filter_images_without_labels=filter_images_without_labels)

    egestor.egest(image_detections=image_detections, root=to_path)
    return True, ''


def validate_helper(image_detection, i):
    try:
        validate_schema(image_detection, IMAGE_DETECTION_SCHEMA)
    except SchemaError as se:
        print(image_detection)
        raise Exception(f"at index {i}") from se

    image = image_detection['image']
    img = cv2.imread(image['path'], 1)
    height, width = img.shape[:2]
    # scale=3
    # extended =(scale-1)/scale/2

    for detection in image_detection['detections']:
        if detection['right'] >= image['width'] or detection['bottom'] >= image['height']:
            print (detection['right'], image['width'],
                   detection['bottom'], image['height'])
            raise ValueError(f"Image {image} has out of bounds bounding box {detection}")
        if detection['right'] <= detection['left'] or detection['bottom'] <= detection['top']:
            raise ValueError(f"Image {image} has zero dimension bbox {detection}")

            # xmin=detection['left']+int(extended*width)
            # ymin=detection['top']+int(extended*height)
            # xmax=detection['right']+int(extended*width)
            # ymax=detection['bottom']+int(extended*height)

        xmin=detection['left']
        ymin=detection['top']
        xmax=detection['right']
        ymax=detection['bottom']

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # if len(image_detection['detections'])<4:
        #     print(image['path'])
        #     # cv2.imshow('image',img)
        #     cv2.waitKey()
    cv2.imwrite('./temp/'+image['path'].split('/')[-1], img)


def validate_image_detections(image_detections):
    print ('running validate_image_detections !')
    # for i, image_detection in enumerate(tqdm(image_detections)):
    # validate_helper(image_detection, i)

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:

        future_to_url = {executor.submit(validate_helper, image_detections[i], i): i for i in tqdm(range(len(image_detections)) )}

        kwargs = {
            'total': len(image_detections),
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



def converter(self, i, image_detections, final_image_detections):
    for detection in image_detections[i]['detections']:
        label = detection['label']
        fallback_label = label if not select_only_known_labels else None
        final_label = convert_dict.get(label.lower(), fallback_label)
        if final_label:
            detection['label'] = final_label
            detections.append(detection)
        # image_detection['detections'] = detections
        image_detections[i]['detections'] = detections
        if detections:
            # final_image_detections.append(image_detection)
            final_image_detections.append(image_detections[i])
        elif not filter_images_without_labels:
            # final_image_detections.append(image_detection)
            final_image_detections.append(image_detections[i])


def convert_labels(*, image_detections, expected_labels,
                   select_only_known_labels, filter_images_without_labels):
    convert_dict = {}
    for label, aliases in expected_labels.items():
        convert_dict[label.lower()] = label
        for alias in aliases:
            convert_dict[alias.lower()] = label

    final_image_detections = []
    print (type(image_detections), 'type')
    # for image_detection in image_detections:
    for i in tqdm(range(len(image_detections))):
        # print('converting labels: ')
        detections = []
        for detection in image_detections[i]['detections']:
            label = detection['label']
            fallback_label = label if not select_only_known_labels else None
            final_label = convert_dict.get(label.lower(), fallback_label)
            if final_label:
                detection['label'] = final_label
                detections.append(detection)
        # image_detection['detections'] = detections
        image_detections[i]['detections'] = detections
        if detections:
            # final_image_detections.append(image_detection)
            final_image_detections.append(image_detections[i])
        elif not filter_images_without_labels:
            # final_image_detections.append(image_detection)
            final_image_detections.append(image_detections[i])
        # self.converter(i,image_detections,[]])
    return final_image_detections

    # # for i in tqdm(range(numOfImgs)):
    # #     self.loading(i)
    # print (multiprocessing.cpu_count())
    # with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
    #     future_to_url = {executor.submit(
    #         self.loading, i): i for i in tqdm(range(numOfImgs))}

    #     kwargs = {
    #         'total': len(future_to_url ),
    #         'unit': 'nap',
    #         'unit_scale': True,
    #         'leave': True
    #     }
    #     for future in tqdm(concurrent.futures.as_completed(future_to_url),**kwargs):
    #         url = future_to_url[future]
    #         try:
    #             data = future.result()
    #         except Exception as exc:
    #             print('%r generated an exception: %s' % (url, exc))
    #         else:
    #             # print('%r page is %d bytes' % (url, len(data)))
    #             pass
