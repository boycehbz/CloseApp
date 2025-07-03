# \[CVPR 2025\] Reconstructing Close Human Interaction with Appearance and Proxemics Reasoning

The code for CVPR 2025 paper "Reconstructing Close Human Interaction with Appearance and Proxemics Reasoning"<br>

[Buzhen Huang](http://www.buzhenhuang.com/), [Chen Li](https://chaneyddtt.github.io/), [Chongyang Xu](https://github.com/Wil909), [Dongyue Lu](https://dylanorange.github.io/), [Jinnan Chen](https://jinnan-chen.github.io/), [Yangang Wang](https://www.yangangwang.com/), [Gim Hee Lee](https://www.comp.nus.edu.sg/~leegh/)<br>
\[[Project](http://www.buzhenhuang.com/works/CloseApp.html)\] \[[Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_Reconstructing_Close_Human_Interaction_with_Appearance_and_Proxemics_Reasoning_CVPR_2025_paper.pdf)\] \[[Dataset]()\]

![figure](/assets/teaser.jpg)

<p align="center">
  <img src="assets/04305.gif" width="49%" />
  <img src="assets/04313.gif" width="49%" />
</p>

## Installation 
The code is tested on Ubuntu 22.04 with a single RTX 3090 GPU.
```
conda create -n closeapp python=3.10
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118

```

Then, compile ```diff-gaussian-rasterization```  as in [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) repository.

[SDF loss](https://github.com/penincillin/SDF_ihmr) and [BVH_CUDA](https://github.com/vchoutas/torch-mesh-isect) are required for evaluating penetration.

Download the official SMPL model from [SMPL](https://smpl.is.tue.mpg.de/) and [SMPLify website](http://smplify.is.tuebingen.mpg.de/) and place them in ```data/smpl/smpl```.<br>

Download data from [Baidu Netdisk](https://pan.baidu.com/s/1CDrDpSZTCiz3A9yUNnMiqw?pwd=y1vt) or [Google Drive](https://drive.google.com/file/d/1kTMD6t-GSJ4tHnC2R19RD7dyMWMXwqsv/view?usp=drive_link).

![figure](/assets/pipeline.jpg)



## Demo
```
python train.py -s data/preprocess_data/04305 -m output/04305 --train_stage=1 --save_render --use_appearance --save_params
```


## Citation
If you find this code or dataset useful for your research, please consider citing the paper.
```
@inproceedings{huang2025reconstructing,
  title={Reconstructing Close Human Interaction with Appearance and Proxemics Reasoning},
  author={Huang, Buzhen and Li, Chen and Xu, Chongyang and Lu, Dongyue and Chen, Jinnan and Wang, Yangang and Lee, Gim Hee},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={17475--17485},
  year={2025}
}

@inproceedings{huang2024closely,
  title={Closely interactive human reconstruction with proxemics and physics-guided adaption},
  author={Huang, Buzhen and Li, Chen and Xu, Chongyang and Pan, Liang and Wang, Yangang and Lee, Gim Hee},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1011--1021},
  year={2024}
}

```

## Acknowledgments
Some of the code are based on the following works.<br>
[CloseInt](https://github.com/boycehbz/HumanInteraction)<br>
[GaussianAvatar](https://github.com/aipixel/GaussianAvatar)<br>
[aitviewer](https://github.com/eth-ait/aitviewer)<br>