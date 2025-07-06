import os

import torch
from mmcv.image import imread

from mmdet.apis import init_detector
from mmocr.apis.inference import model_inference
from mmocr.datasets import build_dataset  # noqa: F401
from mmocr.models import build_detector  # noqa: F401

import sys
import glob
import time
import pickle
import numpy as np
from tqdm import tqdm
from table_recognition.utils import detect_visual, end2end_visual, structure_visual, coord_convert, clip_detect_bbox, rectangle_crop_img, delete_invalid_bbox


def build_model(config_file, checkpoint_file):
    device = 'cpu'
    model = init_detector(config_file, checkpoint=checkpoint_file, device=device)

    if model.cfg.data.test['type'] == 'ConcatDataset':
        model.cfg.data.test.pipeline = model.cfg.data.test['datasets'][
            0].pipeline

    return model


class Inference:
    def __init__(self, config_file, checkpoint_file, device=None):
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.model = build_model(config_file, checkpoint_file)

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            # Specify GPU device
            device = torch.device("cuda:{}".format(device))
        self.model.to(device)

    def result_format(self, pred, file_path):
        raise NotImplementedError

    def predict_single_file(self, file_path):
        pass

    def predict_batch(self, imgs):
        pass


class Structure_Recognition(Inference):
    def __init__(self, config_file, checkpoint_file, samples_per_gpu=1):
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        super().__init__(config_file, checkpoint_file)
        self.samples_per_gpu = samples_per_gpu

    def result_format(self, pred, file_path=None):
        pred = pred[0]
        return pred

    # 这里卓明师兄也修改了
    def predict_single_file(self, file_path):
        # numpy inference
        # img = imread(file_path)
        # file_name = os.path.basename(file_path)
        # result = model_inference(self.model, [img], batch_mode=True)
        # result = self.result_format(result, file_path)
        # result_dict = {file_name:result}
        # return result, result_dict

        
        filename = file_path.split('/')[-1].replace("jpg","txt")
        # print("config:",self.config_file)

        # with open("/data/zml/mmocr_WTW_recognition/StructureLabelAddEmptyBbox_test/"+"table_spider_00496_0.txt", 'r', encoding='UTF-8') as f:
        with open("/home/chs/tablemaster-mmocr/"+"table_spider_00496_0.txt", 'r', encoding='UTF-8') as f:
            lines = f.readlines()

        # print(lines[1])
        lines  = lines[1].split('\n')[0]
        # lines = None
        img = imread(file_path)
        # print("img_path:",file_path)
        # print("img.shape:",img.shape)
        file_name = os.path.basename(file_path)
        print(file_name)
        result = model_inference(self.model, [img], lines,batch_mode=True)
        # print("table:",result)
        result = self.result_format(result, file_path)
        # print(result)
        result_dict = {file_name:result}
        return result, result_dict

class Runner:
    def __init__(self, cfg):
        self.structure_master_config = cfg['structure_master_config']
        self.structure_master_ckpt = cfg['structure_master_ckpt']
        self.structure_master_result_folder = cfg['structure_master_result_folder']

        test_folder = cfg['test_folder']

    def init_structure_master(self):
        self.master_structure_inference = \
            Structure_Recognition(self.structure_master_config, self.structure_master_ckpt)

    def release_structure_master(self):
        torch.cuda.empty_cache()
        del self.master_structure_inference

    def do_structure_predict(self, path, is_save=True, gpu_idx=None):
        if isinstance(path, str):
            if os.path.isfile(path):
                all_results = dict()
                print('Single file in structure master prediction ...')
                _, result_dict = self.master_structure_inference.predict_single_file(path)
                all_results.update(result_dict)

            elif os.path.isdir(path):
                all_results = dict()
                print('Folder files in structure master prediction ...')
                search_path = os.path.join(path, '*.jpg')
                files = glob.glob(search_path)
                # files = files[:20]   # 郭沛添加，用来测试使用
                for file in tqdm(files):
                    _, result_dict = self.master_structure_inference.predict_single_file(file)
                    all_results.update(result_dict)

            else:
                raise ValueError

        elif isinstance(path, list):
            all_results = dict()
            print('Chunks files in structure master prediction ...')
            for i, p in enumerate(path):
                _, result_dict = self.master_structure_inference.predict_single_file(p)
                all_results.update(result_dict)
                if gpu_idx is not None:
                    print("[GPU_{} : {} / {}] {} file structure inference. ".format(gpu_idx, i+1, len(path), p))
                else:
                    print("{} file structure inference. ".format(p))

        else:
            raise ValueError

        # save for matcher.
        if is_save:
            if not os.path.exists(self.structure_master_result_folder):
                os.makedirs(self.structure_master_result_folder)

            if not isinstance(path, list):
                save_file = os.path.join(self.structure_master_result_folder, 'structure_master_results.pkl')
            else:
                save_file = os.path.join(self.structure_master_result_folder, 'structure_master_results_{}.pkl'.format(gpu_idx))

            with open(save_file, 'wb') as f:
                pickle.dump(all_results, f)

    def run(self, path):
        # structure master
        self.init_structure_master()
        self.do_structure_predict(path, is_save=True)
        self.release_structure_master()


if __name__ == '__main__':
    cfg = {
        'structure_master_config': '/home/chs/tablemaster-mmocr/configs/textrecog/master/table_master_ConcatLayer_ResnetExtract_Ranger_tabrecset.py',
        'structure_master_ckpt': '/home/chs/tablemaster-mmocr/work_dir_chs_tabrecset0612/epoch_90.pth',
        'structure_master_result_folder': '/home/chs/tablemaster-mmocr/work_dir_chs_tabrecset0612/results_90',
        'test_folder': '/data/chs/tabrecset/test',
    }

    # single gpu device inference
    runner = Runner(cfg)
    runner.run(cfg['test_folder'])
