import numpy as np

CLASS2Id = {
    'wall': 0,
    'floor': 1,
    'cabinet': 2,
    'bed': 3,
    'chair': 4,
    'sofa': 5,
    'table': 6,
    'door': 7,
    'window': 8,
    'bookshelf': 9,
    'picture': 10,
    'counter': 11,
    'desk': 12,
    'curtain': 13,
    'refridgerator': 14,
    'shower curtain': 15,
    'toilet': 16,
    'sink': 17,
    'bathtub': 18,
    'otherfurniture': 19,
}

# Map relevant classes to {0,1,...,19}, and ignored classes to -100
RELEVANT_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
NUM_TOTAL_CLASSES = 150
NUM_CONCERNED_CLASSES = 20
NUM_RELEVANT_CLASSES = len(RELEVANT_CLASSES)
REMAPPED_IGNORE_CLASSES_ID = -100
CLASS_REMAPPER = np.ones(NUM_TOTAL_CLASSES) * REMAPPED_IGNORE_CLASSES_ID
for _, x in enumerate(RELEVANT_CLASSES):
    CLASS_REMAPPER[x] = _

CLASS_LABELS = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
                'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
VALID_CLASS_IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
ID_TO_LABEL = {k: v for k, v in zip(VALID_CLASS_IDS, CLASS_LABELS)}
LABEL_TO_ID = {k: v for k, v in zip(CLASS_LABELS, VALID_CLASS_IDS)}

COLOR20 = np.array(
    [[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
     [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 190],
     [0, 128, 128], [230, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
     [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128]])

COLOR40 = np.array(
    [[88, 170, 108], [174, 105, 226], [78, 194, 83], [198, 62, 165], [133, 188, 52], [97, 101, 219], [190, 177, 52],
     [139, 65, 168], [75, 202, 137], [225, 66, 129],
     [68, 135, 42], [226, 116, 210], [146, 186, 98], [68, 105, 201], [219, 148, 53], [85, 142, 235], [212, 85, 42],
     [78, 176, 223], [221, 63, 77], [68, 195, 195],
     [175, 58, 119], [81, 175, 144], [184, 70, 74], [40, 116, 79], [184, 134, 219], [130, 137, 46], [110, 89, 164],
     [92, 135, 74], [220, 140, 190], [94, 103, 39],
     [144, 154, 219], [160, 86, 40], [67, 107, 165], [194, 170, 104], [162, 95, 150], [143, 110, 44], [146, 72, 105],
     [225, 142, 106], [162, 83, 86], [227, 124, 143]])

SEMANTIC_IDXS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
SEMANTIC_NAMES = np.array(
    ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter',
     'desk', 'curtain', 'refridgerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture'])
CLASS_COLOR = {
    'unannotated': [0, 0, 0],
    'floor': [143, 223, 142],
    'wall': [171, 198, 230],
    'cabinet': [0, 120, 177],
    'bed': [255, 188, 126],
    'chair': [189, 189, 57],
    'sofa': [144, 86, 76],
    'table': [255, 152, 153],
    'door': [222, 40, 47],
    'window': [197, 176, 212],
    'bookshelf': [150, 103, 185],
    'picture': [200, 156, 149],
    'counter': [0, 190, 206],
    'desk': [252, 183, 210],
    'curtain': [219, 219, 146],
    'refridgerator': [255, 127, 43],
    'bathtub': [234, 119, 192],
    'shower curtain': [150, 218, 228],
    'toilet': [0, 160, 55],
    'sink': [110, 128, 143],
    'otherfurniture': [80, 83, 160]
}
SEMANTIC_IDX2NAME = {1: 'wall', 2: 'floor', 3: 'cabinet', 4: 'bed', 5: 'chair', 6: 'sofa', 7: 'table', 8: 'door',
                     9: 'window', 10: 'bookshelf', 11: 'picture',
                     12: 'counter', 14: 'desk', 16: 'curtain', 24: 'refridgerator', 28: 'shower curtain', 33: 'toilet',
                     34: 'sink', 36: 'bathtub', 39: 'otherfurniture'}

LABEL_to_COLOR_and_NAME = {
    0: {"color": np.array([171, 198, 230]), "name": "wall"},
    1: {"color": np.array([143, 223, 142]), "name": "floor"},
    2: {"color": np.array([0, 120, 177]), "name": "cabinet"},
    3: {"color": np.array([255, 188, 126]), "name": "bed"},
    4: {"color": np.array([189, 189, 57]), "name": "chair"},
    5: {"color": np.array([144, 86, 76]), "name": "sofa"},
    6: {"color": np.array([255, 152, 153]), "name": "table"},
    7: {"color": np.array([222, 40, 47]), "name": "door"},
    8: {"color": np.array([197, 176, 212]), "name": "window"},
    9: {"color": np.array([150, 103, 185]), "name": "bookshelf"},
    10: {"color": np.array([200, 156, 149]), "name": "picture"},
    11: {"color": np.array([0, 190, 206]), "name": "counter"},
    12: {"color": np.array([252, 183, 210]), "name": "desk"},
    13: {"color": np.array([219, 219, 146]), "name": "curtain"},
    14: {"color": np.array([255, 127, 43]), "name": "refridgerator"},
    15: {"color": np.array([150, 218, 228]), "name": "shower curtain"},
    16: {"color": np.array([0, 160, 55]), "name": "toilet"},
    17: {"color": np.array([110, 128, 143]), "name": "sink"},
    18: {"color": np.array([234, 119, 192]), "name": "bathtub"},
    19: {"color": np.array([80, 83, 160]), "name": "otherfurniture"},
}
# the same to: LABEL_to_COLOR_and_NAME = {i:{"color": CLASS_COLOR[SEMANTIC_IDX2NAME[name]],"name": SEMANTIC_IDX2NAME[name]} for i, name in enumerate(SEMANTIC_IDX2NAME)}
