# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 14:41:08 2022

@author: 54756
"""
import numpy as np


def get_objects_from_label(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    objects = [Object3d(line) for line in lines]
    return objects


def cls_type_to_id(cls_type):
    type_to_id = {'Vehicle': 1, 'Pedestrian': 2, 'Bicycle': 3}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]


class Object3d(object):
    def __init__(self, line):
        label = line.strip().split(' ')
        self.src = line
        self.cls_type = label[0]
        self.cls_id = cls_type_to_id(self.cls_type)
        self.l = float(label[4])
        self.w = float(label[5])
        self.h = float(label[6])
        self.loc = np.array((float(label[1]), float(label[2]), float(label[3])), dtype=np.float32)
        self.ry = float(label[7])