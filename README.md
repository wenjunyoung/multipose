MultiPose
======
The repo is  multi-pose estimation for CPU,  GPUs are not needed. you can run it in the Intel CPU with WeCam or video file.

Environment
===
The code was tested on Ubuntu 16.04 & Win10, with Python 3.6 and PyTorch. Intel CPUs &OpenVINO are needed for testing.

Quick start
=====
## Installation
1. python >= 3.6
2. pytorch
3. opencv
4. imutils
5. numpy
6. [OpenVINO >= 2020 R1](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/download.html?elq_cid=6658738_ts1602130313059&erpm_id=9756844_ts1602130313059), following [official instruction](https://docs.openvinotoolkit.org/2021.1/index.html).<br/>
    

## RUN
    # USAGE
    python multi_keypoint.py --xml /path/to/your/model/*.xml --bin /path/to/your/model/*.bin --input_video /path/to/your/video --output /path/to/save/video/file
    
tips: before running, please Set up the OpenVINO environment variables   

License
=======
MIT License (refer to the LICENSE file for details).

Citation
======
If you find this project useful for your research, please use the following BibTeX entry.

        @inproceedings{zhou2019objects,
          title={Objects as Points},
          author={Zhou, Xingyi and Wang, Dequan and Kr{\"a}henb{\"u}hl, Philipp},
          booktitle={arXiv preprint arXiv:1904.07850},
          year={2019}
        }
