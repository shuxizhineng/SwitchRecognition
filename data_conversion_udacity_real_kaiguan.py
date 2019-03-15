# -*- coding: UTF-8 -*-
'''
Usage: python data_conversion_udacity_real.py --output_path output_file_name.record
'''

import tensorflow as tf
import yaml
import os
from object_detection_ori.utils import dataset_util
from xml.dom.minidom import parse
import xml.dom.minidom

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

LABEL_DICT = {
    "small_off" : 1,
    "small_on"  : 2,
    "big_off" : 3,
    "big_on" : 4,
}


def getBoxInfoFromXml(filePath, xmlDir, jpgDir, width, height):

    xmins = []
    xmaxs = []

    ymins = []
    ymaxs = []

    classes_text = []
    classes = []

    jpgPathAndName = []

    for filePathName in [filePath]:
        for xmlBox in os.listdir(filePathName + xmlDir + "/"):
            print('filePathName: ' + filePathName)
            print('xmlDir: ' + xmlDir)
            print('xmlBox: ' + xmlBox)
            print('file name: ' + filePathName + xmlDir + "/" + xmlBox)
            # 使用minidom解析器打开 XML 文档
            DOMTree = xml.dom.minidom.parse(filePathName + xmlDir + "/" + xmlBox)
            collection = DOMTree.documentElement

            xmins.append(float(collection.getElementsByTagName("xmin")[0].childNodes[0].data) / width)
            xmaxs.append(float(collection.getElementsByTagName("xmax")[0].childNodes[0].data) / width)
            ymins.append(float(collection.getElementsByTagName("ymin")[0].childNodes[0].data) / height)
            ymaxs.append(float(collection.getElementsByTagName("ymax")[0].childNodes[0].data) / height)
            classes_text.append(collection.getElementsByTagName("name")[0].childNodes[0].data.encode())
            classes.append(int(LABEL_DICT[collection.getElementsByTagName("name")[0].childNodes[0].data]))
            jpgPathAndName.append(filePathName + jpgDir + "/"+ xmlBox.split(".")[0] + ".jpg")
    #print xmins, xmaxs, ymins, ymaxs, classes_text, classes, jpgPathAndName
    return xmins, xmaxs, ymins, ymaxs, classes_text, classes, jpgPathAndName


def create_tf_example(xmins, xmaxs, ymins, ymaxs, classes_text, classes, jpgPathAndName, width, height):

    filename = jpgPathAndName  # Filename of the image. Empty if image is not from file
    filename = filename.encode()

    with tf.gfile.GFile(jpgPathAndName, 'rb') as fid:
        encoded_image = fid.read()

    image_format = 'jpg'.encode()

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature([xmins]),
        'image/object/bbox/xmax': dataset_util.float_list_feature([xmaxs]),
        'image/object/bbox/ymin': dataset_util.float_list_feature([ymins]),
        'image/object/bbox/ymax': dataset_util.float_list_feature([ymaxs]),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature([classes]),
    }))

    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    filePath = "/home/arthur/SwitchRecognition/data/"
    xmlDir = "on_off_xml_data"
    jpgDir = "on_off_jpg_data"
    height = 480  # Image height
    width = 640  # Image width
    xmins, xmaxs, ymins, ymaxs, classes_text, classes, jpgPathAndName = getBoxInfoFromXml(filePath, xmlDir, jpgDir, width, height)

    len_jpgs = len(jpgPathAndName)
    print("Loaded ", len(jpgPathAndName), "jpgs")

    counter = 0
    for i in range(len(jpgPathAndName)):
        tf_example = create_tf_example(xmins[i], xmaxs[i], ymins[i], ymaxs[i], classes_text[i], classes[i], jpgPathAndName[i], width, height)
        writer.write(tf_example.SerializeToString())

        if counter % 10 == 0:
            print("Percent done", (counter / float(len_jpgs)) * 100)
        counter += 1

    writer.close()


if __name__ == '__main__':
    tf.app.run()

