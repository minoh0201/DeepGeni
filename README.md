# DeepGeni

Deep generalized interpretable autoencoder that learns a latent representation of microbiome profiles for better prediction of immune checkpoint inhibitors response and extracts the most informative microbial taxa involved in modulation of the response

![image](https://user-images.githubusercontent.com/19569318/196018569-80e936ed-fb30-4a48-ae11-adb787a39efa.png)

## Starting guide
Require GPU machine, git, Anaconda3, docker, docker image of `tensorflow/tensorflow:1.13.2-gpu-py3-jupyter`

1. Run docker image equipped with tensorflow:1.13.2-gpu
```
docker run -it --rm -p 8888:8888 -v ~/[DEEPBIOGEN DIR NAME]:/tf/[DEEPBIOGEN DIR NAME] --runtime=nvidia tensorflow/tensorflow:1.13.2-gpu-py3-jupyter bash
```

1. Clone the repo into your local directory
```
git clone [REPO URL]
```
2. Create a virtual environment
```
conda create -n deepgeni python=3.6
```
3. Activate the virtual environment
```
conda activate deepgeni
```
4. Install required packages
```
pip install -r requirements.txt
```
5. Run DeepBioGen and check usage
```
python main.py -h
```

## Experiment guide

1. Run baseline (No FS) experiment
```
python main.py --expname Baseline
```

2. Run Feature-selection-only (FS only) experiment
```
python main.py --expname FS
```

3. Run Feature selection and auto-encoding (FS + AE) experiment
```
python main.py --expname FS-AE
```

4. Run DeepGeni (FS + DBG + AE) experiment
```
python main.py --expname FS-DBG-AE
```

## Citation
TBA

