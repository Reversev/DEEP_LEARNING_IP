# The Usage of Pytorch_lightning 

- 1. Requirement

```bash
bash requirements.txt
```

- 2. Usage

```python
python train.py
```

- 3. Results

The total cost time (including initialize and train):  fp16:107s; fp32:124s

![Train Results](https://github.com/Reversev/DEEP_LEARNING_IP/blob/main/pytorch_lightning/assert/train.png) ![Validation Results](https://github.com/Reversev/DEEP_LEARNING_IP/blob/main/pytorch_lightning/assert/val.png)

- 4. Prediction

![Prediction result](https://github.com/Reversev/DEEP_LEARNING_IP/blob/main/pytorch_lightning/assert/res.png)

```
GroundTruth:   ants  ants  ants  ants  ants  ants  ants  ants
Predicted:   bees  ants  ants  ants  ants  ants  bees  bees
```
