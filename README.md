# FCNet 

![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic)
![pytorch 1.7.0](https://img.shields.io/badge/pytorch-1.7.0-green.svg?style=plastic)

![overview](.\asset\overview.png)

FCNet encompasses three colorization approaches, *i.e.*, single- or multi-reference image-guided colorization in (a), sample-guided colorization in (b), and automatic colorization in (c). In (a), the first column is the grayscale input, and the subsequent five columns show the reference and the corresponding single-reference colorization results. Then, the last column is a multi-reference colorization result, taking colors from different facial components of the references. In (b), the results are generated according to the sampled single or multiple-color representations. In (c), we give our results under automatic settings and the results of competing methods.

![overall_structure](.\asset\architecture.png)

Overview structure of our proposed FCNet: Two main components *g* and the colorization network *f*.

## Dependencies and Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.7](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

1. Clone the repo

   ```bash
   git clone https://github.com/HyZhu39/FCNet.git
   ```

1. Install Dependencies

   ```bash
   cd FCNet
   pip install -r requirements.txt
   ```

## Get Started

### Test

We provide quick test code with the pretrained model. 

1. Download this repo, as well as the pretrained models from [Google Drive](https://drive.google.com/drive/folders/1NOuGxU8ReEJW860SYkGE4n5G47-7JJ9W?usp=drive_link) or [Baidu Drive](https://pan.baidu.com/s/1cohFgBInFZ7dMJE1EHV66w?pwd=FCNT), and unzip.

2. Modify the paths to the test dataset and pretrained model in the following test scripts for configuration.

   ```
   ./scripts/demo_phase1.py
   ./scripts/demo_phase1_mul.py
   ./scripts/demo_phase2.py
   ./scripts/demo_phase3.py
   ```

3. Run test code for *single* reference image-guided colorization.

   ```
   python scripts/demo_phase1.py
   ```

4. Run test code for *sample-guided* colorization.

   ```
   python scripts/demo_phase2.py
   ```

5. Run test code for *automatic* colorization.

   ```
   python scripts/demo_phase3.py
   ```

6. Check out the results in `./results`.

### Train

1. Prepare datasets.

1. Modify the config file and train script.

   ```
   ./training/config.py
   ./scripts/train.py
   ```

1. Run training code. 

## Results

<img src=".\asset\ref-img.png" alt="ref-img" style="zoom:120%;" />
Result of *single- or multi-reference image-guided* colorization.

<img src=".\asset\sampling.png" alt="sampling" style="zoom:120%;" />

Result of *single- or multi-reference image-guided* colorization.

<img src=".\asset\automatic.png" alt="automatic" style="zoom:120%;" />

Result of *automatic* colorization.
