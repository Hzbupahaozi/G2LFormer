from mmdet.datasets.builder import DATASETS
from mmocr.core.evaluation.ocr_metric import eval_ocr_metric
from mmocr.datasets.base_dataset import BaseDataset

import numpy as np
@DATASETS.register_module()
class OCRDataset(BaseDataset):

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['img_info']['ann_file'] = self.ann_file
        results['text'] = results['img_info']['text']
        # print("pre?")

    def evaluate(self, results, metric='acc', logger=None, **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            dict[str: float]
        """
        # print(self.data_infos[0])
        # print(self.ann_file)
        # print(self.img_prefix)
        # print(results)
        gt_texts = []
        pred_texts = []
        for i in range(len(self)):
            item_info = self.data_infos[i]
            filename = item_info['filename']
            text = item_info['text']
            gt_texts.append(text)
            print(text)
            pred_texts.append(results[filename]['text'])
        # eval_results = eval_hmean(
        #     results,
        #     img_infos,
        #     ann_infos,
        #     metrics=metrics,
        #     score_thr=score_thr,
        #     logger=logger,
        #     rank_list=rank_list)
        # eval_results = eval_ocr_metric(pred_texts, gt_texts)

        # return eval_results
    # def evaluate_bbox(self,
    #             results,
    #             metric='bbox',
    #             logger=None,
    #             jsonfile_prefix=None,
    #             classwise=False,
    #             proposal_nums=(100, 300, 1000),
    #             iou_thrs=None,
    #             metric_items=None):
    #     """Evaluation in COCO protocol.

    #     Args:
    #         results (list[list | tuple]): Testing results of the dataset.
    #         metric (str | list[str]): Metrics to be evaluated. Options are
    #             'bbox', 'segm', 'proposal', 'proposal_fast'.
    #         logger (logging.Logger | str | None): Logger used for printing
    #             related information during evaluation. Default: None.
    #         jsonfile_prefix (str | None): The prefix of json files. It includes
    #             the file path and the prefix of filename, e.g., "a/b/prefix".
    #             If not specified, a temp file will be created. Default: None.
    #         classwise (bool): Whether to evaluating the AP for each class.
    #         proposal_nums (Sequence[int]): Proposal number used for evaluating
    #             recalls, such as recall@100, recall@1000.
    #             Default: (100, 300, 1000).
    #         iou_thrs (Sequence[float], optional): IoU threshold used for
    #             evaluating recalls/mAPs. If set to a list, the average of all
    #             IoUs will also be computed. If not specified, [0.50, 0.55,
    #             0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
    #             Default: None.
    #         metric_items (list[str] | str, optional): Metric items that will
    #             be returned. If not specified, ``['AR@100', 'AR@300',
    #             'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
    #             used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
    #             'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
    #             ``metric=='bbox' or metric=='segm'``.

    #     Returns:
    #         dict[str, float]: COCO style evaluation metric.
    #     """

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

    #     # result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

   