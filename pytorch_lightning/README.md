# The Usage of Pytorch_lightning 

- Requirement

```bash
bash requirements.txt
```

- Usage

```python
python train.py
```

- Results

The total cost time (including initialize and train):  fp16:107s; fp32:124s

![Train Results](https://github.com/Reversev/DEEP_LEARNING_IP/blob/main/pytorch_lightning/assert/train.png) ![Validation Results](https://github.com/Reversev/DEEP_LEARNING_IP/blob/main/pytorch_lightning/assert/val.png)

- Prediction

![Prediction result](https://github.com/Reversev/DEEP_LEARNING_IP/blob/main/pytorch_lightning/assert/res.png)

```
GroundTruth:   ants  ants  ants  ants  ants  ants  ants  ants
Predicted:   bees  ants  ants  ants  ants  ants  bees  bees
```
