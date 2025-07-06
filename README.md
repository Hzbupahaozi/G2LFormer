  <h1 align="left">G2LFormer</h1>


![G2LFormer∂'s architecture](architecture.png)




### Dependency

* [MMOCR-0.2.0](https://github.com/open-mmlab/mmocr/tree/v0.2.0)
* [MMDetection-2.11.0](https://github.com/open-mmlab/mmdetection/tree/v2.11.0)
* [mmcv-full-1.3.4](https://github.com/open-mmlab/mmcv/tree/v1.3.4)



<!-- GETTING STARTED -->

## Getting Started

### Installation


1. Install pytorch torchvision torchaudio cython mmpycocotools rapidfuzz numpy.
   
   ```sh
   conda create -n G2LFormer python=3.9
   conda activate G2LFormer
   pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
   pip3 install cython==0.29.33 -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
   pip install mmpycocotools
   pip install rapidfuzz==2.15.1
   pip install numpy==1.23.5
   ```

2. Install mmdetection. click [here](https://github.com/open-mmlab/mmdetection/blob/v2.11.0/docs/get_started.md) for details.
   
   ```sh
   # We embed mmdetection-2.11.0 source code into this project.
   # You can cd and install it (recommend).
   cd ./mmdetection-2.11.0
   pip install -v -e .
   ```
   
3. Install mmocr. click [here](https://github.com/open-mmlab/mmocr/blob/main/docs/install.md) for details.

   ```sh
   # install mmocr
   cd {Path to TableMASTER_mmocr}
   pip install -v -e .
   ```

4. Install mmcv-full-1.4.0. click [here](https://github.com/open-mmlab/mmcv) for details.

   ```sh
   pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
   
   # install mmcv-full-1.4.0 with torch version 1.10.0 cuda_version 11.1
   pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
   ```
   
5. Install MultiScaleDeformableAttention

   ```sh
   cd {Path to Deformable/model/ops}
   sh make.sh
   ```





<!-- USAGE EXAMPLES -->

## Usage

### Data preprocess

Run [data_preprocess.py](./table_recognition/data_preprocess.py) to get valid train data. Remember to change the **'raw_img_root'** and **‘save_root’** property of **PubtabnetParser** to your path.

```shell
python ./table_recognition/data_preprocess.py
```

It will about 8 hours to finish parsing 500777 train files. After finishing the train set parsing, change the property of **'split'** folder in **PubtabnetParser** to **'val'** and get formatted val data.

Directory structure of parsed train data is :

```shell
.
├── StructureLabelAddEmptyBbox_train
│   ├── PMC1064074_007_00.txt
│   ├── PMC1064076_003_00.txt
│   ├── PMC1064076_004_00.txt
│   └── ...
├── recognition_train_img
│   ├── 0
│       ├── PMC1064100_007_00_0.png
│       ├── PMC1064100_007_00_10.png
│       ├── ...
│       └── PMC1064100_007_00_108.png
│   ├── 1
│   ├── ...
│   └── 15
├── recognition_train_txt
│   ├── 0.txt
│   ├── 1.txt
│   ├── ...
│   └── 15.txt
├── structure_alphabet.txt
└── textline_recognition_alphabet.txt
```

We also transfer the raw **Pubtanet** data to **Lmdb** files by the script [lmdb_maker.py](https://github.com/JiaquanYe/TableMASTER-mmocr/blob/master/table_recognition/lmdb_maker.py). 

Click [here](https://pan.baidu.com/s/1X3P2zFpEBN1T_r22l_9zSA) to download the **Pubtanet** data **Lmdb** files (code:uxl1)

If you want to train your model via **Lmdb** files, please take a look at the [TableMASTER lmdb config file](https://github.com/JiaquanYe/TableMASTER-mmocr/blob/master/configs/textrecog/master/table_master_lmdb_ResnetExtract_Ranger_0930.py) and [text-line MASTER lmdb config file](https://github.com/JiaquanYe/TableMASTER-mmocr/blob/master/configs/textrecog/master/master_lmdb_ResnetExtra_tableRec_dataset_dynamic_mmfp16.py)

### Train

 Train table structure recognition model, with **TableMASTER**.

   ```shell
   sh ./table_recognition/expr/table_recognition_dist_train.sh
   ```

### Inference

To get final results, firstly, we need to forward the three up-mentioned models, respectively. Secondly, we merge the results by our matching algorithm, to generate the final HTML code.

Models inference. 

   ```shell
   python ./table_recognition/table_inference_chs_{name of dataset}.py
   ```

### Get TEDS score

1. Installation.

   ```shell
   pip install -r ./table_recognition/PubTabNet-master/src/requirements.txt
   ```

2. Get **gtVal.json**.

   ```shell
   python ./table_recognition/get_val_gt.py
   ```

3. Calcutate TEDS score. Before run this script, modify pred file path and gt file path in [mmocr_teds_acc_mp.py](./table_recognition/PubTabNet-master/src/mmocr_teds_acc_mp.py)

   ```shell
   python ./table_recognition/PubTabNet-master/src/mmocr_teds_acc_mp.py
   ```



<!-- Pretrain Model -->

## Pretrained Model

The **TableMASTER** (TableMASTER_maxlength_500) pretrained model. In the validation set, the accuracy is **0.7767**.

[[Google Drive]](https://drive.google.com/file/d/1LSuVQJ0J8WFtXhLfcCKyzGqcCYmcwEk6/view?usp=sharing)

[[BaiduYun Drive]](https://pan.baidu.com/s/1G2tBpycZY6c6wzfE3V9khw) code:**irp6**

<br/>

The table textline detection model **PSENet** pretrained model. 

 [[Google Drive]](https://drive.google.com/file/d/13vni9GH6cxr5jTiOdiRu--Q6AZojB2p2/view?usp=sharing)

[[BaiduYun Drive]](https://pan.baidu.com/s/1fPdkS6iTA8CKmjsQ7noKLw) code:**6b30**


<!-- LICENSE -->

## License

This project is licensed under the MIT License. See LICENSE for more details.



<!-- Citations -->

## Citations

```latex
@article{G2LFormer,
  title={G2LFormer: Global-to-Local Query Enhancement for Robust Table Structure Recognition},
  author={Haosheng Cai and Yang Xue},
  booktitle={ACM MultiMedia},
  year={2025}
}

```

## Acknowledege

This project is borrowed from TableMaster.
