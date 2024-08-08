# Flux.1-schnell example
This is an example of using Flux.1-schnell: https://huggingface.co/black-forest-labs/FLUX.1-schnell

The only real addition to this is the requirements.txt and the change to use cuda. As the example on huggingface is not complete, this helps others get started.

While this will work with cpu only setups, but it is so slow. So very slow. Good luck. I recommend using CUDA if at all possible. 
# Steps

## Use python 3.10

I observe that the flux repo used 3.10, so I didn't want to fight it. You might be able to get away with another version.

You probably should use pyenv, but who am I to judge?

## Install torch

### For People with CUDA
Use `nvcc --version` to determine version of cuda you have, then check the pytorch website (https://pytorch.org/get-started/locally/) for the correct pip install.

For me:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
### For People who like to suffer
```
pip install torch torchvision torchaudio
```


## Install requirements

```
pip install -r requirements.txt
```

## Run

```
python main.py
```

## Note on my actual steps:

Obviously, i had to make my own requirements.txt, so this is the order I did things.

```
pip install accelerate
pip install sentencepiece
pip install numpy
pip install protobuf
nvcc --version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers
pip install git+https://github.com/huggingface/diffusers.git
```
