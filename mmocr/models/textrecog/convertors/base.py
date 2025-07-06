from mmocr.models.builder import CONVERTORS
from mmocr.utils import list_from_file
import torch

@CONVERTORS.register_module()
class BaseConvertor:
    """Convert between text, index and tensor for text recognize pipeline.

    Args:
        dict_type (str): Type of dict, should be either 'DICT36' or 'DICT90'.
        dict_file (None|str): Character dict file path. If not none,
            the dict_file is of higher priority than dict_type.
        dict_list (None|list[str]): Character list. If not none, the list
            is of higher priority than dict_type, but lower than dict_file.
    """
    start_idx = end_idx = padding_idx = 0
    unknown_idx = None
    lower = False

    DICT36 = tuple('0123456789abcdefghijklmnopqrstuvwxyz')
    DICT90 = tuple('0123456789abcdefghijklmnopqrstuvwxyz'
                   'ABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()'
                   '*+,-./:;<=>?@[\\]_`~')

    def __init__(self, dict_type='DICT90', dict_file=None, dict_list=None):
        assert dict_type in ('DICT36', 'DICT90')
        assert dict_file is None or isinstance(dict_file, str)
        assert dict_list is None or isinstance(dict_list, list)
        self.idx2char = []
        if dict_file is not None:
            for line in list_from_file(dict_file):
                # line = line.strip()
                line = line.strip('\n')  # did not strip space style.
                if line != '':
                    self.idx2char.append(line)
        elif dict_list is not None:
            self.idx2char = dict_list
        else:
            if dict_type == 'DICT36':
                self.idx2char = list(self.DICT36)
            else:
                self.idx2char = list(self.DICT90)

        self.char2idx = {}
        for idx, char in enumerate(self.idx2char):
            self.char2idx[char] = idx

    def num_classes(self):
        """Number of output classes."""
        return len(self.idx2char)

    def str2idx(self, strings):
        """Convert strings to indexes.

        Args:
            strings (list[str]): ['hello', 'world'].
        Returns:
            indexes (list[list[int]]): [[1,2,3,3,4], [5,4,6,3,7]].
        """
        assert isinstance(strings, list)

        indexes = []
        for string in strings:
            if self.lower:
                string = string.lower()
            index = []
            for char in string:
                char_idx = self.char2idx.get(char)
                if char_idx == None:        #hwl gt shaole kongge
                    char = ' ' + char
                    char_idx = self.char2idx.get(char,self.unknown_idx)
                if char_idx is None:
                    raise Exception(f'Chararcter: {char} not in dict,'
                                    f' please check gt_label and use'
                                    f' custom dict file,'
                                    f' or set "with_unknown=True"')
                index.append(char_idx)
            indexes.append(index)

        return indexes

    def str2tensor(self, strings):
        """Convert text-string to input tensor.

        Args:
            strings (list[str]): ['hello', 'world'].
        Returns:
            tensors (list[torch.Tensor]): [torch.Tensor([1,2,3,3,4]),
                torch.Tensor([5,4,6,3,7])].
        """
        raise NotImplementedError

    def idx2str(self, indexes):
        """Convert indexes to text strings.

        Args:
            indexes (list[list[int]]): [[1,2,3,3,4], [5,4,6,3,7]].
        Returns:
            strings (list[str]): ['hello', 'world'].
        """
        assert isinstance(indexes, list)
        strings = []
        for index in indexes:
            string = [self.idx2char[i] for i in index]
            strings.append(''.join(string))

        return strings

    def tensor2idx(self, output):
        """Convert model output tensor to character indexes and scores.
        Args:
            output (tensor): The model outputs with size: N * T * C
        Returns:
            indexes (list[list[int]]): [[1,2,3,3,4], [5,4,6,3,7]].
            scores (list[list[float]]): [[0.9,0.8,0.95,0.97,0.94],
                [0.9,0.9,0.98,0.97,0.96]].
        """
        raise NotImplementedError
    
    ###############################################为了加上噪声所作的处理
    def apply_noise(self, known_labels_expand, label_noise_scale, method):
        if label_noise_scale > 0:
            # 替换和添加的token都是原来的labels里面出现过的
            unique_labels = known_labels_expand.unique()
            if method == 0:
                # 替换操作
                p_replace = torch.rand_like(known_labels_expand.float())
                replace_indice = torch.nonzero(p_replace < label_noise_scale).view(-1)
                # new_label = torch.randint_like(replace_indice, 0, 106)
                # known_labels_expand.scatter_(0, replace_indice, new_label)
                if replace_indice.size(0) > 0:  
                    new_label = unique_labels[torch.randint(0, unique_labels.size(0), (replace_indice.size(0),))]
                    known_labels_expand.scatter_(0, replace_indice, new_label)
                return known_labels_expand

            if method == 1:
                # 删除操作
                p_delete = torch.rand_like(known_labels_expand.float())
                delete_indice = torch.nonzero(p_delete < label_noise_scale).view(-1)
                mask = torch.ones(known_labels_expand.size(0), dtype=torch.bool)
                mask[delete_indice] = False
                known_labels_expand = known_labels_expand[mask]
                return known_labels_expand

            if method == 2:
                # 添加操作
                p_add = torch.rand(known_labels_expand.size(0))
                add_indice = torch.nonzero(p_add < label_noise_scale).view(-1)
                # add_tokens = torch.randint(0, 106, (add_indice.size(0),))
                # add_indice, _ = torch.sort(add_indice)
                if add_indice.size(0) > 0:
                    add_tokens = unique_labels[torch.randint(0, unique_labels.size(0), (add_indice.size(0),))]
                    add_indice, _ = torch.sort(add_indice)

                expanded_size = known_labels_expand.size(0) + add_indice.size(0)
                new_known_labels_expand = torch.empty(expanded_size, dtype=known_labels_expand.dtype)

                current_index = 0
                add_counter = 0
                for i in range(expanded_size):
                    if add_counter < add_indice.size(0) and i == add_indice[add_counter] + add_counter:
                        new_known_labels_expand[i] = add_tokens[add_counter]
                        add_counter += 1
                    else:
                        new_known_labels_expand[i] = known_labels_expand[current_index]
                        current_index += 1
                known_labels_expand = new_known_labels_expand
                return known_labels_expand
    
    ###############################################为了对加噪之后的labels也进行pad填充
    def process_target(self, tensor, start_idx, end_idx, padding_idx, max_seq_len):
        # 创建目标张量并添加起始和结束标记
        src_target = torch.LongTensor(tensor.size(0) + 2).fill_(0)
        src_target[0] = start_idx
        src_target[-1] = end_idx
        src_target[1:-1] = tensor

        # 初始化填充的目标张量
        padded_target = (torch.ones(max_seq_len) * padding_idx).long()
        char_num = src_target.size(0)
        
        # 如果字符数超过最大序列长度，则进行截断
        if char_num > max_seq_len:
            padded_target = src_target[:max_seq_len]
        else:
            padded_target[:char_num] = src_target

        return padded_target
