# Inferring uncertainties, priors and costs from human behavior using neural amortized Bayesian actors
(anonymous NeurIPS 2024 submission)

## Requirements

To install requirements:

```
python -m pip install -r requirements.txt
```

## Training

To generate an evaluation dataset used during training, run:

```
python generate_eval_data.py --cost <cost_function_name>
```

To train an action network, run this command:

```
python train.py --cost <cost_function_name>
```

## Inference

To run inference on a simulated dataset, run:

```
python run_inference.py --cost <cost_function_name>
```

## Results
![results](figure.png)
