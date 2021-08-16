# EfficientNet-Train
**Base line from** [EfficientNet PyTorch][https://github.com/lukemelas/EfficientNet-PyTorch.git]

# QucikStart

**First, you should install efficientnet_pytorch**

```python
pip install efficientnet_pytorch
```

**Train**

To run on Imagenet, place your train and val directories in data, and edit your classes json file.

Example commands:

```python
# Train a small pretrained EfficientNet on CPU
python main.py data -a 'efficientnet-b0' --pretrained
```

```
# Train medium EfficientNet on GPU
python main.py data -a 'efficientnet-b3'  --gpu 0 --batch-size 128
```

```
# Train ResNet-50 on GPU
python main.py data  -a 'resnet50' --gpu 0
```

**Run**

You need to recompose part of the code to satisfy your requests.

```python
python run.py
```

