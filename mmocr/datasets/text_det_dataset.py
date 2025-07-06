import numpy as np

from mmdet.datasets.builder import DATASETS
from mmocr.core.evaluation.hmean import eval_hmean
from mmocr.datasets.base_dataset import BaseDataset

from mmocr.core.evaluation  import utils as eval_utils
@DATASETS.register_module()
class TextDetDataset(BaseDataset):

    def _parse_anno_info(self, annotations):
        """Parse bbox and mask annotation.
        Args:
            annotations (dict): Annotations of one image.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, masks_ignore. "masks"  and
                "masks_ignore" are represented by polygon boundary
                point sequences.
        """
        gt_bboxes, gt_bboxes_ignore = [], []
        gt_masks, gt_masks_ignore = [], []
        gt_labels = []
        for ann in annotations['bbox']:
            # if ann.get('iscrowd', False):
                # gt_bboxes_ignore.append(ann)
                # gt_masks_ignore.append(ann.get('segmentation', None))
            # else:
                gt_bboxes.append(ann)
            #     gt_labels.append(ann['category_id'])
            #     gt_masks.append(ann.get('segmentation', None))
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks_ignore=gt_masks_ignore,
            masks=gt_masks)

        return ann

    def prepare_train_img(self, index):
        """Get training data and annotations from pipeline.

        Args:
            index (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """
        img_ann_info = self.data_infos[index]
        img_info = {
            'filename': img_ann_info['file_name'],
            'height': img_ann_info['height'],
            'width': img_ann_info['width']
        }
        ann_info = self._parse_anno_info(img_ann_info['annotations'])
        results = dict(img_info=img_info, ann_info=ann_info)
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        self.pre_pipeline(results)

        return self.pipeline(results)
    

    # def evaluate(self,
    #              results,
    #              metric='bbox',
    #              score_thr=0.3,
    #              rank_list=None,
    #              logger=None,
    #              **kwargs):
    #     metrics = metric if isinstance(metric, list) else [metric]
    #     allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
    #     for metric in metrics:
    #         if metric not in allowed_metrics:
    #             raise KeyError(f'metric {metric} is not supported')
    #     if iou_thrs is None:
    #         iou_thrs = np.linspace(
    #             .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
    #     if metric_items is not None:
    #         if not isinstance(metric_items, list):
    #             metric_items = [metric_items]

    #     result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

    #     eval_results = OrderedDict()
    #     cocoGt = self.coco
    #     for metric in metrics:
    #         msg = f'Evaluating {metric}...'
    #         if logger is None:
    #             msg = '\n' + msg
    #         print_log(msg, logger=logger)

    #         if metric == 'proposal_fast':
    #             ar = self.fast_eval_recall(
    #                 results, proposal_nums, iou_thrs, logger='silent')
    #             log_msg = []
    #             for i, num in enumerate(proposal_nums):
    #                 eval_results[f'AR@{num}'] = ar[i]
    #                 log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
    #             log_msg = ''.join(log_msg)
    #             print_log(log_msg, logger=logger)
    #             continue

    #         if metric not in result_files:
    #             raise KeyError(f'{metric} is not in results')
    #         try:
    #             cocoDt = cocoGt.loadRes(result_files[metric])
    #         except IndexError:
    #             print_log(
    #                 'The testing results of the whole dataset is empty.',
    #                 logger=logger,
    #                 level=logging.ERROR)
    #             break

    #         iou_type = 'bbox' if metric == 'proposal' else metric
    #         cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
    #         cocoEval.params.catIds = self.cat_ids
    #         cocoEval.params.imgIds = self.img_ids
    #         cocoEval.params.maxDets = list(proposal_nums)
    #         cocoEval.params.iouThrs = iou_thrs
    #         # mapping of cocoEval.stats
    #         coco_metric_names = {
    #             'mAP': 0,
    #             'mAP_50': 1,
    #             'mAP_75': 2,
    #             'mAP_s': 3,
    #             'mAP_m': 4,
    #             'mAP_l': 5,
    #             'AR@100': 6,
    #             'AR@300': 7,
    #             'AR@1000': 8,
    #             'AR_s@1000': 9,
    #             'AR_m@1000': 10,
    #             'AR_l@1000': 11
    #         }
    #         if metric_items is not None:
    #             for metric_item in metric_items:
    #                 if metric_item not in coco_metric_names:
    #                     raise KeyError(
    #                         f'metric item {metric_item} is not supported')

    #         if metric == 'proposal':
    #             cocoEval.params.useCats = 0
    #             cocoEval.evaluate()
    #             cocoEval.accumulate()
    #             cocoEval.summarize()
    #             if metric_items is None:
    #                 metric_items = [
    #                     'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
    #                     'AR_m@1000', 'AR_l@1000'
    #                 ]

    #             for item in metric_items:
    #                 val = float(
    #                     f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
    #                 eval_results[item] = val
    #         else:
    #             cocoEval.evaluate()
    #             cocoEval.accumulate()
    #             cocoEval.summarize()
    #             if classwise:  # Compute per-category AP
    #                 # Compute per-category AP
    #                 # from https://github.com/facebookresearch/detectron2/
    #                 precisions = cocoEval.eval['precision']
    #                 # precision: (iou, recall, cls, area range, max dets)
    #                 assert len(self.cat_ids) == precisions.shape[2]

    #                 results_per_category = []
    #                 for idx, catId in enumerate(self.cat_ids):
    #                     # area range index 0: all area ranges
    #                     # max dets index -1: typically 100 per image
    #                     nm = self.coco.loadCats(catId)[0]
    #                     precision = precisions[:, :, idx, 0, -1]
    #                     precision = precision[precision > -1]
    #                     if precision.size:
    #                         ap = np.mean(precision)
    #                     else:
    #                         ap = float('nan')
    #                     results_per_category.append(
    #                         (f'{nm["name"]}', f'{float(ap):0.3f}'))

    #                 num_columns = min(6, len(results_per_category) * 2)
    #                 results_flatten = list(
    #                     itertools.chain(*results_per_category))
    #                 headers = ['category', 'AP'] * (num_columns // 2)
    #                 results_2d = itertools.zip_longest(*[
    #                     results_flatten[i::num_columns]
    #                     for i in range(num_columns)
    #                 ])
    #                 table_data = [headers]
    #                 table_data += [result for result in results_2d]
    #                 table = AsciiTable(table_data)
    #                 print_log('\n' + table.table, logger=logger)

    #             if metric_items is None:
    #                 metric_items = [
    #                     'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
    #                 ]

    #             for metric_item in metric_items:
    #                 key = f'{metric}_{metric_item}'
    #                 val = float(
    #                     f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
    #                 )
    #                 eval_results[key] = val
    #             ap = cocoEval.stats[:6]
    #             eval_results[f'{metric}_mAP_copypaste'] = (
    #                 f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
    #                 f'{ap[4]:.3f} {ap[5]:.3f}')
    #     if tmp_dir is not None:
    #         tmp_dir.cleanup()
    #     return eval_results

    def evaluate(self,
                 results,
                 metric='hmean-iou',
                 score_thr=0.3,
                 rank_list=None,
                 logger=None,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            score_thr (float): Score threshold for prediction map.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            rank_list (str): json file used to save eval result
                of each image after ranking.
        Returns:
            dict[str: float]
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['hmean-iou', 'hmean-ic13']
        metrics = set(metrics) & set(allowed_metrics)

        img_infos = []
        ann_infos = []
        result = []
        for item in results:
            result.append(results[item])
        print(len(result))
        # print(result)
        # results = result
        dataset_hit_num = 0
        dataset_gt_num = 0
        dataset_pred_num = 0
        d = dict()
        
        for i in range(len(self)):
            img_ann_info = self.data_infos[i]
            # print(img_ann_info)
            img_info = {'filename': img_ann_info['filename']}
            ann_info = self._parse_anno_info(img_ann_info)
            # print(img_ann_info)
            filename = img_ann_info['filename']
            # print(filename)
            # print(results[filename]['bbox'])
            # print(ann_info['bboxes'])
            if(filename not in results):continue
            gt_polys = [eval_utils.box2polygon(p) for p in ann_info['bboxes']]
            # print(gt_polys)
            det_polys = [eval_utils.box2polygon(p) for p in results[filename]['bbox'][1]]
            # print(det_polys)
            length = min(len(gt_polys), len(det_polys))
            text = img_ann_info['text'].split(',')
            # print(results[filename]['text'])
            pre_text = results[filename]['text'][0].split(',')
            # print(pre_text)
            iou_thr = 0.5
            gt_hit = 0
            pred_hit = 0
            hit_num = 0
            l  = 0
            for i in range(length):
                if((text[i]== '<td></td>' or text[i]== '<td') ):
                    gt_hit += 1
                    if( text[i] == pre_text[i]):
                        iou_mat = eval_utils.poly_iou(gt_polys[i], det_polys[i])
                        if iou_mat > iou_thr:   
                            hit_num += 1
                if((pre_text[i]== '<td></td>' or pre_text[i]== '<td') ):
                    pred_hit += 1
                    # print(i,iou_mat)
            img_infos.append(img_info)
            ann_infos.append(ann_info)
            p,r = hit_num/pred_hit, hit_num/gt_hit
            # if(hit_num==0): f=0
            # else: f = 2*p*r/(p+r)
            r, p, f = eval_utils.compute_hmean(hit_num, hit_num, gt_hit,
                                           pred_hit)
            # print(r,p,h)
            d[filename] = (p,r,f)
            dataset_hit_num += hit_num
            dataset_gt_num += gt_hit
            dataset_pred_num += pred_hit
        # dataset_p,dataset_r= dataset_hit_num/dataset_pred_num, dataset_hit_num/dataset_gt_num
        # dataset_h= 2*dataset_p*dataset_r/(dataset_p+dataset_r)
        dataset_r, dataset_p, dataset_h = eval_utils.compute_hmean(
            dataset_hit_num, dataset_hit_num, dataset_gt_num, dataset_pred_num)
        print( dataset_p,dataset_r, dataset_h )
        import json
        with open("result.json","w") as f:
            json.dump(d,f)
        # eval_results = eval_hmean(
        #     results,
        #     img_infos,
        #     ann_infos,
        #     metrics=metrics,
        #     score_thr=score_thr,
        #     logger=logger,
        #     rank_list=rank_list)

        # return eval_results
