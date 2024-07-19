# wflow-TartanVO: Attenuation-Aware Weighted Optical Flow with Medium Transmission Map for Learning-based Visual Odometry in Underwater terrain

This paper addresses the challenge of improving learning-based monocular visual odometry (VO) in underwater environments by integrating principles of underwater optical imaging to manipulate optical flow estimation. Leveraging the inherent properties of underwater imaging, the novel wflow-TartanVO is introduced, enhancing the accuracy of VO systems for autonomous underwater vehicles (AUVs). The proposed method utilizes a normalized medium transmission map as a weight map to adjust the estimated optical flow for emphasizing regions with lower degradation and suppressing uncertain regions affected by underwater light scattering and absorption. wflow-TartanVO does not require fine-tuning of pre-trained VO models, thus promoting its adaptability to different environments and camera models. Evaluation of different real-world underwater datasets demonstrates the outperformance of wflow-TartanVO over baseline VO methods, as evidenced by the considerably reduced Absolute Trajectory Error (ATE).

Please check out our [Paper](https://arxiv.org/abs/2407.13159). 


## Setting up the environment 

- install conda
- create environment from .yml file: `$ conda env create -f environment.yml`

Our code has been tested on Ubuntu 22.04, and a Cuda version of 11.8 


## Testing with a pretrained model

### Download the testing data
  
* Download Aqualoc-Seq9 testing trajectory
URL: https://drive.google.com/drive/folders/18GwbbX8pxkZRqzw6TDCokio9PoSkxnDp?usp=drive_link

Download data and put in folder: `data/Aqualoc/seq9`

* Download SubPipe-Chunk3 testing trajectory
URL: https://drive.google.com/drive/folders/1EdjxXY-_tsto6kpYmvlhoF05A33-UvuJ?usp=sharing

Download data and put in folder: `data/SubPipe/chunk3`


### Run the testing script

- Testing on Aqualoc-Seq9

```
# TartanVO
$ python vo_trajectory_from_folder.py --model-name tartanvo_1914.pkl --aqualoc --batch-size 1 --worker-num 1 --test-dir data/Aqualoc/seq9/images_sequence_9_3000 --pose-file data/Aqualoc/seq9/poses_colmap_inverse_3000.txt 
```

```
# wflow-TartanVO
$ python test_uie.py --model-name tartanvo_1914.pkl --uie-model-name model_best_val_2062.pth.tar --aqualoc --batch-size 1 --worker-num 1 --test-dir data/Aqualoc/seq9/images_sequence_9_3000 --pose-file data/Aqualoc/seq9/poses_colmap_inverse_3000.txt
```

- Testing on SubPipe-chunk3

```
# TartanVO
$ python vo_trajectory_from_folder.py --model-name tartanvo_1914.pkl --subpipe --batch-size 1 --worker-num 1 --test-dir data/SubPipe/chunk3/Cam0_images --pose-file data/SubPipe/chunk3/poses.txt
```

```
# wflow-TartanVO
$ python test_uie_rgb.py --model-name tartanvo_1914.pkl --uie-model-name model_best_val_2062.pth.tar --subpipe --batch-size 1 --worker-num 1 --test-dir data/SubPipe/chunk3/Cam0_images --pose-file data/SubPipe/chunk3/poses.txt
```


## Paper

Please cite this as:

```
@article{wflowtartanvo2024,
  title={Attenuation-Aware Weighted Optical Flow with Medium Transmission Map for Learning-based Visual Odometry in Underwater terrain},
  author={Nguyen Gia Bach, Chanh Minh Tran, Eiji Kamioka, and Phan Xuan Tan},
  journal={2024 IEEE 7th International Conference on Multimedia Information Processing and Retrieval (MIPR)},
  year={2024},
  pages={1-4}
}
```
