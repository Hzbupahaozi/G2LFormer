import torch

import mmocr.utils as utils
from mmocr.models.builder import CONVERTORS
from .base import BaseConvertor
import os
import numpy as np

@CONVERTORS.register_module()
class MasterConvertor(BaseConvertor):
    """Convert between text, index and tensor for encoder-decoder based
    pipeline.

    Args:
        dict_type (str): Type of dict, should be one of {'DICT36', 'DICT90'}.
        dict_file (None|str): Character dict file path. If not none,
            higher priority than dict_type.
        dict_list (None|list[str]): Character list. If not none, higher
            priority than dict_type, but lower than dict_file.
        with_unknown (bool): If True, add `UKN` token to class.
        max_seq_len (int): Maximum sequence length of label.
        lower (bool): If True, convert original string to lower case.
        start_end_same (bool): Whether use the same index for
            start and end token or not. Default: True.
    """

    def __init__(self,
                 dict_type='DICT90',
                 dict_file=None,
                 dict_list=None,
                 with_unknown=True,
                 max_seq_len=40,
                 lower=False,
                 start_end_same=True,
                 **kwargs):
        super().__init__(dict_type, dict_file, dict_list)
        assert isinstance(with_unknown, bool)
        assert isinstance(max_seq_len, int)
        assert isinstance(lower, bool)

        self.with_unknown = with_unknown
        self.max_seq_len = max_seq_len
        self.lower = lower
        self.start_end_same = start_end_same

        self.update_dict()

    def update_dict(self):
        start_token = '<SOS>'
        end_token = '<EOS>'
        unknown_token = '<UKN>'
        padding_token = '<PAD>'

        # unknown
        self.unknown_idx = None
        if self.with_unknown:
            self.idx2char.append(unknown_token)
            self.unknown_idx = len(self.idx2char) - 1

        # SOS/EOS
        self.idx2char.append(start_token)
        self.start_idx = len(self.idx2char) - 1
        if not self.start_end_same:
            self.idx2char.append(end_token)
        self.end_idx = len(self.idx2char) - 1

        # padding
        self.idx2char.append(padding_token)
        self.padding_idx = len(self.idx2char) - 1

        # update char2idx
        self.char2idx = {}
        for idx, char in enumerate(self.idx2char):
            self.char2idx[char] = idx
    """
    def str2tensor(self, strings):
        # ordinary OCR task strings is list of str, but table master is list of list.
        assert utils.is_type_list(strings, str) or utils.is_type_list(strings, list)

        tensors, padded_targets = [], []
        padded_labels_expand1s, padded_labels_expand2s, padded_labels_expand3s = [], [], []
        indexes = self.str2idx(strings) # 里面有3个，每个都是没有加上开头结尾的token
        for index in indexes:
            tensor = torch.LongTensor(index)
            tensors.append(tensor)
            #####################################################################为了label的加噪加上的
            scalar = 3
            label_noise_scale = 0.1
            known_labels = tensor.repeat(scalar,1)
            known_labels_expand1 = self.apply_noise(known_labels[0].clone(), label_noise_scale, method=0)
            known_labels_expand2 = self.apply_noise(known_labels[1].clone(), label_noise_scale, method=1)
            known_labels_expand3 = self.apply_noise(known_labels[2].clone(), label_noise_scale, method=2)
            
            #####################################################################

            #####################################################################因为这里要对加噪之后的token进行pad填充所以也写成了一个函数
            padded_target = self.process_target(tensor, self.start_idx, self.end_idx, self.padding_idx, self.max_seq_len)
            padded_labels_expand1 = self.process_target(known_labels_expand1, self.start_idx, self.end_idx, self.padding_idx, self.max_seq_len)
            padded_labels_expand2 = self.process_target(known_labels_expand2, self.start_idx, self.end_idx, self.padding_idx, self.max_seq_len)
            padded_labels_expand3 = self.process_target(known_labels_expand3, self.start_idx, self.end_idx, self.padding_idx, self.max_seq_len)

            # target tensor for loss
            # src_target = torch.LongTensor(tensor.size(0) + 2).fill_(0)
            # src_target[-1] = self.end_idx
            # src_target[0] = self.start_idx
            # src_target[1:-1] = tensor   # torch.size([68])   
            # padded_target = (torch.ones(self.max_seq_len) *
            #                  self.padding_idx).long()   # [600]每个都是108即padding的token
            # char_num = src_target.size(0)
            # if char_num > self.max_seq_len:
            #     # TODO:大于max_seq_len-2的，应该跳过？检查dataset有没处理。
            #     padded_target = src_target[:self.max_seq_len]
            # else:
            #     # TODO:这里是最后一个是PAD token，而不是EOS，与FASTOCR不同，其最后一个是EOS.
            #     padded_target[:char_num] = src_target
            #########################################################################
            padded_targets.append(padded_target)
            padded_labels_expand1s.append(padded_labels_expand1)
            padded_labels_expand2s.append(padded_labels_expand2)
            padded_labels_expand3s.append(padded_labels_expand3)
        padded_targets = torch.stack(padded_targets, 0).long()
        padded_labels_expand1s = torch.stack(padded_labels_expand1s, 0).long()
        padded_labels_expand2s = torch.stack(padded_labels_expand2s, 0).long()
        padded_labels_expand3s = torch.stack(padded_labels_expand3s, 0).long()

        return {'targets': tensors, 'padded_targets': padded_targets, 'padded_labels_expand1s': padded_labels_expand1s, 'padded_labels_expand2s': padded_labels_expand2s, 'padded_labels_expand3s': padded_labels_expand3s}
    """
    def str2tensor(self, strings):
        """
        Convert text-string into tensor.
        Args:
            strings (list[str]): ['hello', 'world']
        Returns:
            dict (str: Tensor | list[tensor]):
                tensors (list[Tensor]): [torch.Tensor([1,2,3,3,4]),
                                                    torch.Tensor([5,4,6,3,7])]
                padded_targets (Tensor(bsz * max_seq_len))
        """
        # ordinary OCR task strings is list of str, but table master is list of list.
        assert utils.is_type_list(strings, str) or utils.is_type_list(strings, list)

        tensors, padded_targets = [], []
        indexes = self.str2idx(strings)
        for index in indexes:
            tensor = torch.LongTensor(index)
            tensors.append(tensor)
            # target tensor for loss
            src_target = torch.LongTensor(tensor.size(0) + 2).fill_(0)
            src_target[-1] = self.end_idx
            src_target[0] = self.start_idx
            src_target[1:-1] = tensor
            padded_target = (torch.ones(self.max_seq_len) *
                             self.padding_idx).long()
            char_num = src_target.size(0)
            if char_num > self.max_seq_len:
                # TODO:大于max_seq_len-2的，应该跳过？检查dataset有没处理。
                padded_target = src_target[:self.max_seq_len]
            else:
                # TODO:这里是最后一个是PAD token，而不是EOS，与FASTOCR不同，其最后一个是EOS.
                padded_target[:char_num] = src_target
            padded_targets.append(padded_target)
        padded_targets = torch.stack(padded_targets, 0).long()

        return {'targets': tensors, 'padded_targets': padded_targets}    

    def tensor2idx(self, outputs, img_metas=None):
        """
        Convert output tensor to text-index
        Args:
            outputs (tensor): model outputs with size: N * T * C
            img_metas (list[dict]): Each dict contains one image info.
        Returns:
            indexes (list[list[int]]): [[1,2,3,3,4], [5,4,6,3,7]]
            scores (list[list[float]]): [[0.9,0.8,0.95,0.97,0.94],
                                         [0.9,0.9,0.98,0.97,0.96]]
        """
        batch_size = outputs.size(0)
        ignore_indexes = [self.padding_idx]
        indexes, scores = [], []
        for idx in range(batch_size):
            seq = outputs[idx, :, :]
            seq = seq.softmax(-1)
            max_value, max_idx = torch.max(seq, -1)
            str_index, str_score = [], []
            output_index = max_idx.cpu().detach().numpy().tolist()
            output_score = max_value.cpu().detach().numpy().tolist()
            for char_index, char_score in zip(output_index, output_score):
                if char_index in ignore_indexes:
                    continue
                if char_index == self.end_idx:
                    break
                str_index.append(char_index)
                str_score.append(char_score)

            indexes.append(str_index)
            scores.append(str_score)

        return indexes, scores
    def tensor2idx_span(self, col_span, img_metas=None):
        """
        Convert output tensor to text-index
        Args:
            outputs (tensor): model outputs with size: N * T * C
            img_metas (list[dict]): Each dict contains one image info.
        Returns:
            indexes (list[list[int]]): [[1,2,3,3,4], [5,4,6,3,7]]
            scores (list[list[float]]): [[0.9,0.8,0.95,0.97,0.94],
                                         [0.9,0.9,0.98,0.97,0.96]]
        """
        batch_size = col_span.size(0)
        ignore_indexes = [self.padding_idx]
        indexes, scores = [], []
        for idx in range(batch_size):
            seq = col_span[idx, :, :]
            seq = seq.softmax(-1)
            max_value, max_idx = torch.max(seq, -1)
            str_index, str_score = [], []
            output_index = max_idx.cpu().detach().numpy().tolist()
            output_score = max_value.cpu().detach().numpy().tolist()
            for char_index, char_score in zip(output_index, output_score):
                if char_index in ignore_indexes:
                    continue
                if char_index == self.end_idx:
                    break
                str_index.append(char_index)
                str_score.append(char_score)

            indexes.append(str_index)
            scores.append(str_score)

        return indexes, scores

@CONVERTORS.register_module()
class TableMasterConvertor(MasterConvertor):
    """Similarity with MasterConvertor, but add key 'bbox' and 'bbox_masks'.
    'bbox' and 'bbox_mask' need to the same length as 'text'.
    This convert use the alphabet extract by data_preprocess.py of table_recognition.

    Args:
        dict_type (str): Type of dict, should be one of {'DICT36', 'DICT90'}.
        dict_file (None|str): Character dict file path. If not none,
            higher priority than dict_type.
        dict_list (None|list[str]): Character list. If not none, higher
            priority than dict_type, but lower than dict_file.
        with_unknown (bool): If True, add `UKN` token to class.
        max_seq_len (int): Maximum sequence length of label.
        lower (bool): If True, convert original string to lower case.
        start_end_same (bool): Whether use the same index for
            start and end token or not. Default: True.
    """
    def __init__(self,
                 dict_type='DICT90',
                 dict_file=None,
                 dict_list=None,
                 with_unknown=True,
                 max_seq_len=500,
                 lower=False,
                 start_end_same=False,
                 **kwargs
                 ):
        self.start_end_same = start_end_same
        self.checker()
        super().__init__(dict_type, dict_file, dict_list, with_unknown, max_seq_len, lower, start_end_same)
        # self.deal_alphabet_span_token()

    def checker(self):
        try:
            assert self.start_end_same is False
        except AssertionError:
            raise

    # def deal_alphabet_span_token(self):
    #     """
    #     Modify the self.idx2char in base, which read by alphabet file.
    #     Reading alphabet will strip space char in the head, eg. ' colspan' -> 'colspan'.
    #     This function will modify self.idx2char and self.char2idx,
    #     to add space char in span-style after reading alphabet.
    #
    #     PS:
    #         If use line.strip('\n') in reading alphabet file in base.py, comment this function.
    #     :return:
    #     """
    #     # modify idx2char
    #     new_alphabet = []
    #     for char in self.idx2char:
    #         char = char.replace('colspan=', ' colspan=')
    #         char = char.replace('rowspan=', ' rowspan=')
    #         new_alphabet.append(char)
    #     self.idx2char = new_alphabet
    #     # modify char2idx
    #     new_dict = {}
    #     for idx, char in enumerate(self.idx2char):
    #         new_dict[char] = idx
    #     self.char2idx = new_dict
    #     import pdb;pdb.set_trace()

    def _pad_bbox(self, bboxes):
        padded_bboxes = []
        for bbox in bboxes:
            bbox = torch.from_numpy(bbox)
            bbox_pad = torch.Tensor([0., 0., 0., 0.]).float()
            padded_bbox = torch.zeros(self.max_seq_len, 4)
            padded_bbox[:] = bbox_pad
            if bbox.shape[0] > self.max_seq_len - 2:
                # case sample's length over max_seq_len
                padded_bbox[1:self.max_seq_len-1] = bbox[:self.max_seq_len-2]
            else:
                padded_bbox[1:len(bbox)+1] = bbox
            padded_bboxes.append(padded_bbox)
        padded_bboxes = torch.stack(padded_bboxes, 0).float()
        return padded_bboxes

    def _pad_bbox_mask(self, bbox_masks):
        padded_bbox_masks = []
        for bbox_mask in bbox_masks:
            bbox_mask = torch.from_numpy(bbox_mask)
            bbox_mask_pad = torch.Tensor([0])
            padded_bbox_mask = torch.zeros(self.max_seq_len)
            padded_bbox_mask[:] = bbox_mask_pad
            if bbox_mask.shape[0] > self.max_seq_len - 2:
                # case sample's length over max_seq_len
                padded_bbox_mask[1:self.max_seq_len-1] = bbox_mask[:self.max_seq_len-2]
            else:
                padded_bbox_mask[1:len(bbox_mask)+1] = bbox_mask
            padded_bbox_masks.append(padded_bbox_mask)
        padded_bbox_masks = torch.stack(padded_bbox_masks, 0).long()
        return padded_bbox_masks

    def idx2str(self, indexes):
        """
        Similar with the 'idx2str' function of base, but use ',' to join the token list.
        :param indexes: (list[list[int]]): [[1,2,3,3,4], [5,4,6,3,7]].
        :return:
        """
        assert isinstance(indexes, list)
        strings = []
        for index in indexes:
            string = [self.idx2char[i] for i in index]
            # use ',' to join char list.
            string = ','.join(string)
            strings.append(string)

        return strings
    def idx2str_span(self, indexes, col_span, row_span):
        """
        Similar with the 'idx2str' function of base, but use ',' to join the token list.
        :param indexes: (list[list[int]]): [[1,2,3,3,4], [5,4,6,3,7]].
        :return:
        """
        assert isinstance(indexes, list)
        strings = []
        for index in indexes:
            colspan_index = np.where(index==4)
            rowspan_index = np.where(index==5)
            string = [self.idx2char[i] for i in index]
            # use ',' to join char list.
            string = ','.join(string)
            strings.append(string)

        return strings

    def _get_pred_bbox_mask(self, strings):
        """
        get the bbox mask by the pred strings results, where 1 means to output.
        <SOS>, <EOS>, <PAD> and <eb></eb> series set to 0, <td></td>, <td set to 1, others set to 0.
        :param strings: pred text list by cls branch.
        :return: pred bbox_mask
        """
        assert isinstance(strings, list)
        pred_bbox_masks = []
        SOS = self.idx2char[self.start_idx]
        EOS = self.idx2char[self.end_idx]
        PAD = self.idx2char[self.padding_idx]

        for string in strings:
            pred_bbox_mask = []
            char_list = string.split(',')
            for char in char_list:
                if char == EOS:
                    pred_bbox_mask.append(0)
                    break
                elif char == PAD:
                    pred_bbox_mask.append(0)
                    continue
                elif char == SOS:
                    pred_bbox_mask.append(0)
                    continue
                else:
                    if char == '<td></td>' or char == '<td':
                        pred_bbox_mask.append(1)
                    else:
                        pred_bbox_mask.append(0)
            pred_bbox_masks.append(pred_bbox_mask)

        return np.array(pred_bbox_masks)

    def _filter_invalid_bbox(self, output_bbox, pred_bbox_mask):
        """
        filter the invalid bboxes, use pred_bbox_masks and value not in [0,1].
        :param output_bbox:
        :param pred_bbox_mask:
        :return:
        """
        # filter bboxes coord out of [0,1]
        low_mask = (output_bbox >= 0.) * 1
        high_mask = (output_bbox <= 1.) * 1
        mask = np.sum((low_mask + high_mask), axis=1)
        # print("mask:",mask)
        value_mask = np.where(mask == 2*4, 1, 0)
        # print("value:",value_mask)
        output_bbox_len = output_bbox.shape[0]
        pred_bbox_mask_len = pred_bbox_mask.shape[0]
        padded_pred_bbox_mask = np.zeros(output_bbox_len, dtype='int64')
        # print(output_bbox_len ,pred_bbox_mask_len)
        pred_bbox_mask_len = min(output_bbox_len ,pred_bbox_mask_len)
        padded_pred_bbox_mask[:pred_bbox_mask_len] = pred_bbox_mask[:pred_bbox_mask_len]
        filtered_output_bbox = \
            output_bbox * np.expand_dims(value_mask, 1) * np.expand_dims(padded_pred_bbox_mask, 1)

        return filtered_output_bbox


    def _decode_bboxes(self, outputs_bbox, pred_bbox_masks, img_metas):
        """
        De-normalize and scale back the bbox coord.
        :param outputs_bbox:
        :param pred_bbox_masks:
        :param img_metas:
        :return:
        """
        pred_bboxes = []
        # print("outputs_bbox:",len(outputs_bbox))
        # print(outputs_bbox)
        n = len(outputs_bbox)
        for i in range(n):
            # for output_bbox,
            #  pred_bbox_mask, img_meta in zip(outputs_bbox[i], pred_bbox_masks, img_metas):
            for output_bbox, pred_bbox_mask, img_meta in zip(outputs_bbox[i], pred_bbox_masks, img_metas):
                output_bbox = output_bbox.cpu().numpy()
                scale_factor = img_meta['scale_factor']
                pad_shape = img_meta['pad_shape']
                ori_shape = img_meta['ori_shape']
                # print(len(output_bbox))

                output_bbox[:] = self._filter_invalid_bbox(output_bbox[:], pred_bbox_mask)
                # print("output:",output_bbox)
                # output_bbox = self._filter_invalid_bbox(output_bbox, pred_bbox_mask)
                # de-normalize to pad shape
                output_bbox[:, 0], output_bbox[:, 2] = output_bbox[:, 0] * pad_shape[1], output_bbox[:, 2] * pad_shape[1]
                output_bbox[:, 1], output_bbox[:, 3] = output_bbox[:, 1] * pad_shape[0], output_bbox[:, 3] * pad_shape[0]
                # output_bbox[:, 0::2] = output_bbox[:, 0::2] * pad_shape[1]
                # output_bbox[:, 1::2] = output_bbox[:, 1::2] * pad_shape[0]

                # scale to origin shape
                output_bbox[:, 0], output_bbox[:, 2] = output_bbox[:, 0] / scale_factor[1], output_bbox[:, 2] / scale_factor[1]
                output_bbox[:, 1], output_bbox[:, 3] = output_bbox[:, 1] / scale_factor[0], output_bbox[:, 3] / scale_factor[0]
                # output_bbox[:, 0::2] = output_bbox[:, 0::2] / scale_factor[1]
                # output_bbox[:, 1::2] = output_bbox[:, 1::2] / scale_factor[0]

                pred_bboxes.append(output_bbox)
        # print("pred:",len(pred_bboxes))
        # print(pred_bboxes)
        return pred_bboxes

    def _adjsut_bboxes_len(self, bboxes, strings):
        # print(len(bboxes),len(bboxes[0]))
        # print("bboxes:",bboxes.shape)
        new_bboxes = []
        # bbox0 = bboxes[0]
        # print("strings:",strings)
        n = len(bboxes)
        for i in range(n):
            for bbox, string in zip([bboxes[i]], strings):
                string = string.split(',')
                string_len = len(string)
                # print(string_len)
                # bbox = bbox[:string_len, :]
                bbox = bbox[:string_len]
                new_bboxes.append(bbox)
        # print(new_bboxes)
        return new_bboxes

    def _get_strings_scores(self, str_scores):
        """
        Calculate strings scores by averaging str_scores
        :param str_scores: softmax score of each char.
        :return:
        """
        # chs修改，因为hwl的数据集会出现分母为0的情况
        # strings_scores = []
        # for str_score in str_scores:
        #     score = sum(str_score) / len(str_score)
        #     strings_scores.append(score)
        # return strings_scores

        strings_scores = []
        for str_score in str_scores:
            if not str_score:
                score = 0
                print('score is 0')
            else:
                score = sum(str_score) / len(str_score)
                print('score:', score)
            strings_scores.append(score)
        return strings_scores

    def _get_filename(self, img_metas):
        filenames = []
        for meta in img_metas:
            filename = os.path.basename(meta.get('filename'))
            filenames.append(filename)
        return filenames
    
    def str_format(self, img_metas):
        """
        Convert text-string into tensor.
        Pad 'bbox' and 'bbox_masks' to the same length as 'text'

        Args:
            img_metas (list[dict]):
                dict.keys() ['filename', 'ori_shape', 'img_shape', 'text', 'scale_factor', 'bbox', 'bbox_masks']
        Returns:
            dict (str: Tensor | list[tensor]):
                tensors (list[Tensor]): [torch.Tensor([1,2,3,3,4]),
                                                    torch.Tensor([5,4,6,3,7])]
                padded_targets (Tensor(bsz * max_seq_len))

                bbox (list[Tensor]):
                bbox_masks (Tensor):
        """

        # output of original str2tensor function(split by ',' in each string).
        # print("img_meta:",img_metas)
        gt_labels = [[char for char in img_meta['text'].split(',')] for img_meta in img_metas]
        tmp_dict = self.str2tensor(gt_labels)
        text_target = tmp_dict['targets']
        text_padded_target = tmp_dict['padded_targets']

        # pad bbox_mask's length
        bbox_masks = [img_meta['bbox_masks'] for img_meta in img_metas]
        bbox_masks = self._pad_bbox_mask(bbox_masks)

        format_dict = {'targets': text_target,
                        'padded_targets': text_padded_target,
                        'bbox_masks': bbox_masks
                        }

        return format_dict 

    def generate_span_label(self,img_metas):
        spanrow_labels = []
        spancol_labels = []
        padded_rows = []
        padded_cols = []
        for img_meta in img_metas:
            spanrow_label = torch.LongTensor([item[0] for item in img_meta["cls_bbox"]])
            spancol_label = torch.LongTensor([item[1] for item in img_meta["cls_bbox"]])
            padded_row = (torch.ones(599) *
                             0).long()
            padded_col = (torch.ones(599) *
                             0).long()
            char_num = len(spanrow_label)
            char_num = min(599,char_num)
            padded_row[:char_num] = spanrow_label
            padded_col[:char_num] = spancol_label
            spanrow_labels.append(spanrow_label)
            spancol_labels.append(spancol_label)
            padded_rows.append(padded_row)
            padded_cols.append(padded_col)
            # print(img_meta["text"])
            # print("pad:",padded_row[:char_num],padded_col[:char_num])
        padded_cols = torch.stack(padded_cols, 0)   # 其实就是打包成[3,599]
        padded_rows = torch.stack(padded_rows, 0)
        return  padded_cols,padded_rows

    def compute_average_heightandwight(self,gt_labels,text_target,bboxes):
        batch=  len(text_target)
        row_value,col_value = [],[]
        pos_list = []
        # print("b:",batch)
        for b in range(batch):
            html =text_target[b]
            row,col = dict(),dict()
            id_list = []
            l = len(text_target[b])
            r,c = 0,0
            for i in range(l):
                if(html[i]==2):
                    r+=1
                    c= 0
                if(html[i]==1):
                    if(r not in row): row[r]=[bboxes[b,i+1]] 
                    else: row[r].append(bboxes[b,i+1])
                    if(c not in col): col[c]=[bboxes[b,i+1]]
                    else: col[c].append(bboxes[b,i+1])
                    c+=1
                if(html[i]==3):
                    rowspan,colspan = 1,1 
                    str1 = gt_labels[b][i+1].split("\"")
                    if( str1[0]=="colspan="):
                        colspan = int(str1[1])
                        str2 = gt_labels[b][i+2].split("\"")
                        if( str2[0]!="rowspan="):
                            if(r not in row): row[r]=[bboxes[b,i+1]] 
                            else: row[r].append(bboxes[b,i+1])
                    elif(str1[0]=="rowspan="):
                        rowspan = int(str1[1])
                        str2 = gt_labels[b][i+2].split("\"")
                        if( str2[0]=="colspan="):
                            colspan = int(str2[1])
                        else:
                            if(c not in col): col[c]=[bboxes[b,i+1]]
                            else: col[c].append(bboxes[b,i+1])  
                    c +=colspan
                    # r +=rowspan
            
            for item in row:
                # print(row[item])
                row_height = sum(row[item])/len(row[item])
                # print("row:",row_height)
                row[item] = row_height[3] 
            for item in col:
                col_weight = sum(col[item])/len(col[item])
                col[item] = col_weight[2]    
            row_value.append(row) 
            col_value.append(col)    
        for b in range(batch):
            html =text_target[b]
            l = len(text_target[b])
            r,c = 0,0
            id_list = []
            for i in range(l):
                if(html[i]==2):
                    r+=1
                    c= 0
                    id_list.append([0,0]) 
                elif(html[i]==1):
                    id_list.append([r,c])
                    c+=1
                elif(html[i]==3):
                    rowspan,colspan = 1,1 
                    str1 = gt_labels[b][i+1].split("\"")
                    if( str1[0]=="colspan="):
                        colspan = int(str1[1])
                    elif(str1[0]=="rowspan="):
                        rowspan = int(str1[1])
                    str2 = gt_labels[b][i+2].split("\"")
                    if( str2[0]=="colspan="):
                        colspan = int(str2[1])
                    elif(str2[0]=="rowspan="):
                        rowspan = int(str2[1])
                    id_list.append([r,c])  
                    # print(r,c,colspan)  
                    c +=colspan
                else: id_list.append([0,0])
            pos_list.append(id_list)
        # print("done")
        return pos_list,row_value,col_value
    def str_bbox_format(self, img_metas):
        """
        Convert text-string into tensor.
        Pad 'bbox' and 'bbox_masks' to the same length as 'text'

        Args:
            img_metas (list[dict]):
                dict.keys() ['filename', 'ori_shape', 'img_shape', 'text', 'scale_factor', 'bbox', 'bbox_masks']
        Returns:
            dict (str: Tensor | list[tensor]):
                tensors (list[Tensor]): [torch.Tensor([1,2,3,3,4]),
                                                    torch.Tensor([5,4,6,3,7])]
                padded_targets (Tensor(bsz * max_seq_len))

                bbox (list[Tensor]):
                bbox_masks (Tensor):
        """

        # output of original str2tensor function(split by ',' in each string).
        

        gt_labels = [[char for char in img_meta['text'].split(',')] for img_meta in img_metas]  # 每张图片的text标注，然后用,分开得到['<tr>','<td></td>''...]        
         
        tmp_dict = self.str2tensor(gt_labels)
        text_target = tmp_dict['targets']   # index之后的标注信息
        text_padded_target = tmp_dict['padded_targets']     # 加了开头、结尾、padding信息的index

        # pad bbox's length
        bboxes = [img_meta['bbox'] for img_meta in img_metas]
        bboxes = self._pad_bbox(bboxes) # 这里是从原来的框信息变成了500个框信息

        padded_cols,padded_rows = self.generate_span_label(img_metas)   # TODO: 这一部分内容是多的
        # padded_cols,padded_rows = None,None
        pos_list,avg_row,avg_col =  self.compute_average_heightandwight(gt_labels,text_target,bboxes)
        # pos_list,avg_row,avg_col =  None,None,None

        # pad bbox_mask's length
        bbox_masks = [img_meta['bbox_masks'] for img_meta in img_metas]
        bbox_masks = self._pad_bbox_mask(bbox_masks)    # 这里同理，mask是01值代表有没有表格，然后pad到500
        # print('padded_targets:',text_padded_target[:,:25])
        # print('bbox_masks:',bbox_masks [:,:25])
        # text_padded_target = None
        # bboxes = None
        # print('targets:',text_target[0][:25])
        
        # 为了加入ccm_loss加入的
        # ccm_targets = []
        # ccm_params = [20,100,400]
        # for i in range(len(img_metas)):
        #     tgt_num = img_metas[i]['num_cell']
        #     t = 0
        #     for j in range(len(ccm_params)):
        #         if tgt_num >= ccm_params[j]:
        #             t = j+1
        #     ccm_targets.append(t)
        # ccm_targets = torch.tensor(ccm_targets)
        # format_dict = {'targets': text_target,
        #                 'padded_targets': text_padded_target,
        #                 'bbox': bboxes,
        #                 'bbox_masks': bbox_masks,
        #                 'cls_bbox': [padded_rows,padded_cols],
        #                 "pos":pos_list,
        #                 "avg_row": avg_row,
        #                 "avg_col": avg_col,
        #                 "num_cell":ccm_targets
        #                 }


        format_dict = {'targets': text_target,
                'padded_targets': text_padded_target,
                'bbox': bboxes,
                'bbox_masks': bbox_masks,
                'cls_bbox': [padded_rows,padded_cols],
                "pos":pos_list,
                "avg_row": avg_row,
                "avg_col": avg_col
                }

        return format_dict

    def output_format(self, outputs, out_bbox, img_metas=None):
        # print("format")
        # cls_branch process
        str_indexes, str_scores = self.tensor2idx(outputs, img_metas)
        strings = self.idx2str(str_indexes)
        scores = self._get_strings_scores(str_scores)

        # bbox_branch process
        pred_bbox_masks = self._get_pred_bbox_mask(strings)
        # import numpy as np
        # pred_bbox_masks = np.ones((1, 599))
        pred_bboxes = self._decode_bboxes(out_bbox, pred_bbox_masks, img_metas)
        # print(len(pred_bboxes))
        # print(len(pred_bboxes[0]))
        pred_bboxes = self._adjsut_bboxes_len(pred_bboxes, strings)

        return strings, scores, pred_bboxes
    
    def output_format_train(self, outputs, img_metas=None):
        # print("format")
        # cls_branch process
        str_indexes, str_scores = self.tensor2idx(outputs, img_metas)
        strings = self.idx2str(str_indexes)
        scores = self._get_strings_scores(str_scores)
        filenames = self._get_filename(img_metas)

        return strings, scores, filenames
