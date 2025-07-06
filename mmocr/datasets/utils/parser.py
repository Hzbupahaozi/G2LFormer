import json
import os
from mmocr.datasets.builder import PARSERS
from mmocr.utils import convert_bbox
import numpy as np
import cv2

@PARSERS.register_module()
class LineStrParser:
    """Parse string of one line in annotation file to dict format.

    Args:
        keys (list[str]): Keys in result dict.
        keys_idx (list[int]): Value index in sub-string list
            for each key above.
        separator (str): Separator to separate string to list of sub-string.
    """

    def __init__(self,
                 keys=['filename', 'text'],
                 keys_idx=[0, 1],
                 separator=' '):
        assert isinstance(keys, list)
        assert isinstance(keys_idx, list)
        assert isinstance(separator, str)
        assert len(keys) > 0
        assert len(keys) == len(keys_idx)
        self.keys = keys
        self.keys_idx = keys_idx
        self.separator = separator

    def get_item(self, data_ret, index):
        map_index = index % len(data_ret)
        line_str = data_ret[map_index]
        for split_key in self.separator:
            if split_key != ' ':
                line_str = line_str.replace(split_key, ' ')
        line_str = line_str.split()
        if len(line_str) <= max(self.keys_idx):
            raise Exception(
                f'key index: {max(self.keys_idx)} out of range: {line_str}')

        line_info = {}
        for i, key in enumerate(self.keys):
            line_info[key] = line_str[self.keys_idx[i]]
        return line_info


@PARSERS.register_module()
class TableTextLineStrParser:
    """Parse string of one line in annotation file to dict format.

    Args:
        keys (list[str]): Keys in result dict.
        keys_idx (list[int]): Value index in sub-string list
            for each key above.
        separator (str): Separator to separate string to list of sub-string.
    """

    def __init__(self,
                 keys=['filename', 'text'],
                 keys_idx=[0, 1],
                 separator=' '):
        assert isinstance(keys, list)
        assert isinstance(keys_idx, list)
        assert isinstance(separator, str)
        assert len(keys) > 0
        assert len(keys) == len(keys_idx)
        self.keys = keys
        self.keys_idx = keys_idx
        self.separator = separator

    def get_item(self, data_ret, index):
        map_index = index % len(data_ret)
        line_str = data_ret[map_index]
        line_str_part = []
        line_str = line_str.split(self.separator)
        line_str_part.append(line_str[0])  # file_path
        # line_str_part.append(''.join(line_str[1:]))  # merge text_list
        # remove the space char at begin of text by strip.
        line_str_part.append(''.join(line_str[1:]).strip())

        if len(line_str_part) <= max(self.keys_idx):
            raise Exception(
                f'key index: {max(self.keys_idx)} out of range: {line_str}')

        line_info = {}
        for i, key in enumerate(self.keys):
            line_info[key] = line_str_part[self.keys_idx[i]]
        return line_info


@PARSERS.register_module()
class LineJsonParser:
    """Parse json-string of one line in annotation file to dict format.

    Args:
        keys (list[str]): Keys in both json-string and result dict.
    """

    def __init__(self, keys=[], **kwargs):
        assert isinstance(keys, list)
        assert len(keys) > 0
        self.keys = keys

    def get_item(self, data_ret, index):
        map_index = index % len(data_ret)
        line_json_obj = json.loads(data_ret[map_index])
        line_info = {}
        for key in self.keys:
            if key not in line_json_obj:
                raise Exception(f'key {key} not in line json {line_json_obj}')
            line_info[key] = line_json_obj[key]

        return line_info


# some functions for table structure label parse.
def build_empty_bbox_mask(bboxes):
    """
    Generate a mask, 0 means empty bbox, 1 means non-empty bbox.
    :param bboxes: list[list] bboxes list
    :return: flag matrix.
    """
    flag = [1 for _ in range(len(bboxes))]
    for i, bbox in enumerate(bboxes):
        # empty bbox coord in label files
        if bbox == [0,0,0,0]:
            flag[i] = 0
    return flag

def get_bbox_nums_by_text(text):
    text = text.split(',')
    pattern = ['<td></td>', '<td', '<eb></eb>',
               '<eb1></eb1>', '<eb2></eb2>', '<eb3></eb3>',
               '<eb4></eb4>', '<eb5></eb5>', '<eb6></eb6>',
               '<eb7></eb7>', '<eb8></eb8>', '<eb9></eb9>',
               '<eb10></eb10>']
    count = 0
    for t in text:
        if t in pattern:
            count += 1
    return count

def align_bbox_mask(bboxes, empty_bbox_mask, label):
    """
    This function is used to in insert [0,0,0,0] in the location, which corresponding
    structure label is non-bbox label(not <td> style structure token, eg. <thead>, <tr>)
    in raw label file. This function will not insert [0,0,0,0] in the empty bbox location,
    which is done in label-preprocess.

    :param bboxes: list[list] bboxes list
    :param empty_bboxes_mask: the empty bbox mask
    :param label: table structure label
    :return: aligned bbox structure label
    """
    pattern = ['<td></td>', '<td', '<eb></eb>',
               '<eb1></eb1>', '<eb2></eb2>', '<eb3></eb3>',
               '<eb4></eb4>', '<eb5></eb5>', '<eb6></eb6>',
               '<eb7></eb7>', '<eb8></eb8>', '<eb9></eb9>',
               '<eb10></eb10>']
    # print(len(bboxes),get_bbox_nums_by_text(label),len(empty_bbox_mask))
    assert len(bboxes) == get_bbox_nums_by_text(label) == len(empty_bbox_mask)
    bbox_count = 0
    structure_token_nums = len(label.split(','))
    # init with [0,0,0,0], and change the real bbox to corresponding value
    aligned_bbox = [[0., 0., 0., 0.] for _ in range(structure_token_nums)]
    aligned_empty_bbox_mask = [1 for _ in range(structure_token_nums)]
    cls_bbox = [ [0,0] for _ in range(structure_token_nums)]
    if(len(bboxes[0])==6):
        flag = 1
    else: flag =0 
    for idx, l in enumerate(label.split(',')):
        if l in pattern:
            if(flag):
                aligned_bbox[idx] = bboxes[bbox_count][:-2]
                cls_bbox[idx] = bboxes[bbox_count][-2:]
            else:
                aligned_bbox[idx] = bboxes[bbox_count]
            aligned_empty_bbox_mask[idx] = empty_bbox_mask[bbox_count]
            bbox_count += 1
    return aligned_bbox, aligned_empty_bbox_mask,cls_bbox

def build_bbox_mask(label):
    #TODO : need to debug to keep <eb></eb> or not.
    structure_token_nums = len(label.split(','))
    pattern = ['<td></td>', '<td', '<eb></eb>', '<tr>', '</tr>']
    mask = [0 for _ in range(structure_token_nums)]
    for idx, l in enumerate(label.split(',')):
        if l in pattern:
           mask[idx] = 1
    return np.array(mask)


def build_tr_mask(label):
    #TODO : need to debug to keep <eb></eb> or not.
    tr_structure_token_nums = len(label.split(','))
    tr_pattern = ['<tr>', '</tr>']
    tr_mask = [0 for _ in range(tr_structure_token_nums)]
    for idx, l in enumerate(label.split(',')):
        if l in tr_pattern:
           tr_mask[idx] = 1
    return np.array(tr_mask)

def build_col_mask(label):
    #TODO : need to debug to keep <eb></eb> or not.
    col_structure_token_nums = len(label.split(','))
    col_pattern = [' colspan']
    col_mask = [0 for _ in range(col_structure_token_nums)]
    for idx, l in enumerate(label.split(',')):
        if l in col_pattern:
           col_mask[idx-1] = 1  # 这里-1是为了能够和cls_box对应上
    return np.where(np.array(col_mask)==1)[0]

def build_row_mask(label):
    #TODO : need to debug to keep <eb></eb> or not.
    row_structure_token_nums = len(label.split(','))
    row_pattern = [' rowspan']
    row_mask = [0 for _ in range(row_structure_token_nums)]
    for idx, l in enumerate(label.split(',')):
        if l in row_pattern:
           row_mask[idx-1] = 1
    return np.where(np.array(row_mask)==1)[0]

# 第二个版本，这里每一行的最大最小x值是根据部分的矩阵的坐标来决定的
def generate_cell_masks2(tr_masks, bboxes):
    # 找出非全0的方框索引
    indices = np.where(tr_masks == 1)[0]
    bboxes_mask_col = np.copy(bboxes)
    bboxes_mask_row = np.copy(bboxes)
    for i in range(0, len(indices)-1,2):
        start_tr_index = indices[i]
        end_tr_index = indices[i+1]
        sub_bboxes_col = bboxes_mask_col[start_tr_index:end_tr_index+1]
        # sub_bboxes_row = bboxes_mask_row[start_tr_index:end_tr_index+1]
        non_zero_indices_col = (sub_bboxes_col != 0).any(axis=1)
        min_x1 = np.min(sub_bboxes_col[non_zero_indices_col, 0])
        max_x2 = np.max(sub_bboxes_col[non_zero_indices_col, 2])
        # min_y1 = np.min(sub_bboxes_row[non_zero_indices, 1])
        # max_y2 = np.min(sub_bboxes_row[non_zero_indices, 3])
        sub_bboxes_col[non_zero_indices_col, 0] = min_x1
        sub_bboxes_col[non_zero_indices_col, 2] = max_x2
        # sub_bboxes_row[non_zero_indices, 1] = min_y1
        # sub_bboxes_row[non_zero_indices, 3] = max_y2
        bboxes_mask_col[start_tr_index:end_tr_index+1] = sub_bboxes_col
        # bboxes_mask_row[start_tr_index:end_tr_index+1] = sub_bboxes_row

    non_zero_indices_row = (bboxes_mask_row != 0).any(axis = 1)

    # # 计算所有非全0方框的 x1 和 x2 的最小值和最大值
    # min_x1 = np.min(bboxes_mask_col[non_zero_indices, 0])
    # max_x2 = np.max(bboxes_mask_col[non_zero_indices, 2])
    min_y1 = np.min(bboxes_mask_row[non_zero_indices_row, 1])
    max_y2 = np.max(bboxes_mask_row[non_zero_indices_row, 3])

    # # 修改非全0方框的 x1 和 x2
    # bboxes_mask_col[non_zero_indices, 0] = min_x1
    # bboxes_mask_col[non_zero_indices, 2] = max_x2
    bboxes_mask_row[non_zero_indices_row, 1] = min_y1
    bboxes_mask_row[non_zero_indices_row, 3] = max_y2
    return bboxes_mask_col, bboxes_mask_row

# 第一个版本 里面有一个最大值和最小值搞错了
def generate_cell_masks(bboxes):
    # 找出非全0的方框索引
    bboxes_mask_col = np.copy(bboxes)
    bboxes_mask_row = np.copy(bboxes)
    non_zero_indices = (bboxes != 0).any(axis=1)

    # 计算所有非全0方框的 x1 和 x2 的最小值和最大值
    min_x1 = np.min(bboxes_mask_col[non_zero_indices, 0])
    max_x2 = np.max(bboxes_mask_col[non_zero_indices, 2])
    min_y1 = np.min(bboxes_mask_row[non_zero_indices, 1])
    max_y2 = np.min(bboxes_mask_row[non_zero_indices, 3])

    # 修改非全0方框的 x1 和 x2
    bboxes_mask_col[non_zero_indices, 0] = min_x1
    bboxes_mask_col[non_zero_indices, 2] = max_x2
    bboxes_mask_row[non_zero_indices, 1] = min_y1
    bboxes_mask_row[non_zero_indices, 3] = max_y2
    return bboxes_mask_col, bboxes_mask_row

def generate_tr_boxes_mask(tr_masks, bboxes):
    indices = np.where(tr_masks == 1)[0]
    output_tr_bboxes = np.zeros_like(bboxes)
    for i in range(0, len(indices)-1,2):
        start_tr_index = indices[i]
        end_tr_index = indices[i+1]
        sub_bboxes = bboxes[start_tr_index:end_tr_index+1]
        sub_bboxes = sub_bboxes[~np.all(sub_bboxes == 0, axis=1)]
        min_x1 = np.min(sub_bboxes[:,0])
        min_y1 = np.min(sub_bboxes[:,1])
        max_x2 = np.max(sub_bboxes[:,2])
        min_y2 = np.min(sub_bboxes[:,3])    # 这里取最小的原因是因为同一行的单元格的纵坐标可能不是一样的，比如说第一个单元格是跨行的第二个不是，但是隶属同一行
        output_tr_bboxes[start_tr_index] = [min_x1, min_y1, max_x2, min_y2]
        output_tr_bboxes[end_tr_index] = [min_x1, min_y1, max_x2, min_y2]
    return output_tr_bboxes

@PARSERS.register_module()
class TableStrParser:
    """Parse a dict which include 'file_path', 'bbox', 'label' to training dict format.
    The advance parse will do here.

    Args:
        keys (list[str]): Keys in result dict.
        keys_idx (list[int]): Value index in sub-string list
            for each key above.
        separator (str): Separator to separate string to list of sub-string.
    """

    def __init__(self,
                 keys=['filename', 'text'],
                 keys_idx=[0, 1],
                 separator=','):
        assert isinstance(keys, list)
        assert isinstance(keys_idx, list)
        assert isinstance(separator, str)
        assert len(keys) > 0
        assert len(keys) == len(keys_idx)
        self.keys = keys
        self.keys_idx = keys_idx
        self.separator = separator

    def get_item(self, data_ret, index):
        # print("parse")
        map_index = index % len(data_ret)
        line_dict = data_ret[map_index]
        file_name = os.path.basename(line_dict['file_path'])
        # print(file_name)
        text = line_dict['label']
        bboxes = line_dict['bbox']

        # advance parse bbox
        empty_bbox_mask = build_empty_bbox_mask(bboxes)
        # print("parse",file_name)
        bboxes, empty_bbox_mask,cls_bbox = align_bbox_mask(bboxes, empty_bbox_mask, text)
        bboxes = np.array(bboxes)
        empty_bbox_mask = np.array(empty_bbox_mask)

        bbox_masks = build_bbox_mask(text)
        tr_masks = build_tr_mask(text)  # 每个HTML序列中tr和/tr的mask
        col_masks = build_col_mask(text)    # 这里生成的mask实在<td,rowspan,></td>里面的<td
        row_masks = build_row_mask(text)

        bbox_masks = bbox_masks * empty_bbox_mask
        # cell_masks_col, cell_masks_row = generate_cell_masks2(tr_masks, bboxes)    # 版本2：单个cell的一列、一行的bboxes
        # 好像有点影响所以复制一个
        bboxes2 = np.copy(bboxes)
        # 为了加入tr的坐标信息
        tr_boxes = generate_tr_boxes_mask(tr_masks, bboxes2)
        tr_boxes2 = np.copy(tr_boxes)

        cell_masks_col, cell_masks_row = generate_cell_masks(bboxes2)

        line_info = {}
        line_info['filename'] = file_name
        line_info['text'] = text
        line_info['bbox'] = bboxes + tr_boxes
        # 为了DQ模块的计数
        line_info['num_cell'] = len(bboxes)
        line_info['bbox_masks'] = bbox_masks
        line_info["cls_bbox"] = cls_bbox
        # 为了实现曼哈顿的视觉引导模块
        # line_info['cell_masks'] = [cell_masks_col,cell_masks_row, bboxes2]  # 这里的bboxes2是传入原始的坐标以便实现曼哈顿的视觉对齐
        line_info['cell_masks'] = [cell_masks_col, cell_masks_row, tr_boxes2]   # 这里的tr_boxes2是搭配VG3使用的，就是对tr的坐标也计算视觉损失
        # line_info['cell_masks'] = [bboxes, bboxes] # 这里是VAST的实现方式
        line_info['tr_masks'] = tr_masks
        line_info['colrow_masks']=[col_masks, row_masks]
        # line_info['tr_masks'] = None
    
        # print("parse")
        return line_info


@PARSERS.register_module()
class TableMASTERLmdbParser:
    """Parse a dict which include 'file_path', 'bbox', 'label' to training dict format.
    The lmdb's data advance parse will do here.

    Args:
        keys (list[str]): Keys in result dict.
        keys_idx (list[int]): Value index in sub-string list
            for each key above.
        separator (str): Separator to separate string to list of sub-string.
        max_seq_len (int): Max sequence, to filter the samples's label longer than this.
    """

    def __init__(self,
                 keys=['filename', 'text'],
                 keys_idx=[0, 1],
                 separator=',',
                 max_seq_len=40):
        assert isinstance(keys, list)
        assert isinstance(keys_idx, list)
        assert isinstance(separator, str)
        assert len(keys) > 0
        assert len(keys) == len(keys_idx)
        self.keys = keys
        self.keys_idx = keys_idx
        self.separator = separator

    def get_item(self, data_ret, index):
        map_index = index % len(data_ret)
        data = data_ret[map_index]

        # img_name, img, info_lines
        file_name = data[0]
        bytes = data[1]
        buf = np.frombuffer(bytes, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        info_lines = data[2]  # raw data from TableMASTER annotation file.

        # parse info_lines
        raw_data = info_lines.strip().split('\n')
        raw_name, text = raw_data[0], raw_data[1]  # don't filter the samples's length over max_seq_len.
        bbox_str_list = raw_data[2:]
        bbox_split = ','
        bboxes = [convert_bbox(bsl.strip().split(bbox_split)) for bsl in bbox_str_list]

        # advance parse bbox
        empty_bbox_mask = build_empty_bbox_mask(bboxes)
        bboxes, empty_bbox_mask = align_bbox_mask(bboxes, empty_bbox_mask, text)
        bboxes = np.array(bboxes)
        empty_bbox_mask = np.array(empty_bbox_mask)

        bbox_masks = build_bbox_mask(text)
        bbox_masks = bbox_masks * empty_bbox_mask

        line_info = {}
        line_info['filename'] = file_name
        line_info['text'] = text
        line_info['bbox'] = bboxes
        line_info['bbox_masks'] = bbox_masks
        line_info['img'] = img

        return line_info


@PARSERS.register_module()
class MASTERLmdbParser:
    """Parse a dict which include 'file_path', 'bbox', 'label' to training dict format.
    The lmdb's data advance parse will do here.

    Args:
        keys (list[str]): Keys in result dict.
        keys_idx (list[int]): Value index in sub-string list
            for each key above.
        separator (str): Separator to separate string to list of sub-string.
        max_seq_len (int): Max sequence, to filter the samples's label longer than this.
    """

    def __init__(self,
                 keys=['filename', 'text'],
                 keys_idx=[0, 1],
                 separator='\t'):
        # useless for this class object.
        assert isinstance(keys, list)
        assert isinstance(keys_idx, list)
        assert isinstance(separator, str)
        assert len(keys) > 0
        assert len(keys) == len(keys_idx)
        self.keys = keys
        self.keys_idx = keys_idx
        self.separator = separator

    def get_item(self, data_ret, index):
        map_index = index % len(data_ret)
        data = data_ret[map_index]

        # img, label
        bytes = data[0]
        text = data[1]
        buf = np.frombuffer(bytes, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        line_info = {}
        line_info['filename'] = str(map_index)
        line_info['text'] = text
        line_info['img'] = img

        return line_info