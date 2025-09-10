carla_labels = {
    0: ("Unlabeled",        [0, 0, 0]),
    1: ("Building",         [70, 70, 70]),
    2: ("Fence",            [190, 153, 153]),
    3: ("Other",            [250, 170, 160]),
    4: ("Pedestrian",       [220, 20, 60]),
    5: ("Pole",             [153, 153, 153]),
    6: ("RoadLine",         [157, 234, 50]),
    7: ("Road",             [128, 64, 128]),
    8: ("Sidewalk",         [244, 35, 232]),
    9: ("Vegetation",       [107, 142, 35]),
    10: ("Vehicle",         [0, 0, 142]),
    11: ("Wall",            [102, 102, 156]),
    12: ("TrafficSign",     [220, 220, 0]),
    13: ("Sky",             [70, 130, 180]),
    14: ("Ground",          [81, 0, 81]),
    15: ("Bridge",          [150, 100, 100]),
    16: ("RailTrack",       [230, 150, 140]),
    17: ("GuardRail",       [180, 165, 180]),
    18: ("TrafficLight",    [250, 170, 30]),
    19: ("Static",          [110, 190, 160]),
    20: ("Dynamic",         [170, 120, 50]),
    21: ("Water",           [45, 60, 150]),
    22: ("Terrain",         [145, 170, 100]),
}
# Kitti-Labels:
#trainId_to_label = {
 #   0: ("road",          [128, 64,128]),
 #   1: ("sidewalk",      [244, 35,232]),
 #   2: ("building",      [ 70, 70, 70]),
 #   3: ("wall",          [102,102,156]),
 #   4: ("fence",         [190,153,153]),
 #   5: ("pole",          [153,153,153]),
 #   6: ("traffic light", [250,170, 30]),
 #   7: ("traffic sign",  [220,220,  0]),
 #   8: ("vegetation",    [107,142, 35]),
 #   9: ("terrain",       [152,251,152]),
 #   10:("sky",           [ 70,130,180]),
 #   11:("person",        [220, 20, 60]),
 #   12:("rider",         [255,  0,  0]),
 #   13:("car",           [  0,  0,142]),
 #   14:("truck",         [  0,  0, 70]),
 #   15:("bus",           [  0, 60,100]),
 #   16:("train",         [  0, 80,100]),
 #   17:("motorcycle",    [  0,  0,230]),
 #   18:("bicycle",       [119, 11, 32]),
 #   255:("void",         [  0,  0,  0])
#}

kitti_to_carla = {
    0: 7,    # road
    1: 8,    # sidewalk
    2: 1,    # building
    3: 11,   # wall
    4: 2,    # fence
    5: 5,    # pole
    6: 18,   # traffic light
    7: 12,   # traffic sign
    8: 9,    # vegetation
    9: 22,   # terrain
    10: 13,  # sky
    11: 4,   # person
    12: 20,  # rider → dynamic (alternativ: eigene Klasse)
    13: 10,  # car
    14: 10,  # truck → vehicle
    15: 10,  # bus → vehicle
    16: 10,  # train → vehicle
    17: 10,  # motorcycle → vehicle
    18: 10   # bicycle → vehicle
}