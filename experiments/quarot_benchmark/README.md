### QuaRot Benchmarking
Here, we are benchmarking Quarot end-to-end inference framework, and see what is the runtime of the hadamard step v.s. the whole runtime


#### step 1: pull quarot code
```sh
git clone https://github.com/spcl/QuaRot.git
```
#### step 2: quarot setup + build
follow steps on QuaRot codebase or use dockerfile provided here:
1. build docker image and launch docker
```sh
sudo docker build --no-cache -t quarot_image .
sudo docker run --rm -it \
  --gpus all \
  -v ./QuaRot:/workspace/QuaRot \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  quarot_image
```

2. download dependencies (in the docker container)
```sh
pip install .
python setup.py build_ext --inplace
pip install transformers==4.36.0
```

#### step 3: run custom benchmark code
```
python layer_flatten.py
```