# Adversarial Training with Dual Batch Normalization

## Authors

- Annonymized authors

## Abstract

While Adversarial Training remains the standard in improving robustness to adversarial attack, it often sacrifices accuracy on natural (clean) samples to a significant extent. Dual-domain training, optimizing on both clean and adversarial objectives, can help realize a better trade-off between clean accuracy and robustness. In this paper, we develop methods to make this kind of training more effective. We show that existing methods suffer from poor performance due to a poor training procedure and overfitting to a particular attack. Then, we develop novel methods to address these problems. First, we demonstrate that adding a KLD regularization to the dual training objective improves on prior methods, on CIFAR-10 and a 10-class subset of ImageNet. Then, taking inspiration from domain adaptation, we develop a new normalization technique, Dual Batch Normalization (DBN), to further improve both clean and robust performance. Combining these two strategies, our model improves on prior methods for dual-domain adversarial training.

## Requirement

See `environment.yml`.

## How to run

- `train.py`: training script takes path to a config file as an argument
- `test.py`: test script takes path to a config file as an argument
- `config/config_cifar10.yml`: config file for CIFAR-10
- `config/config_imagenette.yml`: config file for Imagenette

Example commands:

```
python train.py config/config_cifar10.yml
python test.py config/config_img.yml
```

## Parameters

- Descriptions and options for each parameter are included in `config/config_cifar10.yml`.
- Parameters marked with `TODO` should be changed to run the desired experiments.
- To use DBN, set both `meta/exp_wrapper: 'match_at'` and `core/normalization_layer: 'dbn'`.
- To use any other dual-domain training, set `meta/exp_wrapper: 'pair_at'` and `core/normalization_layer` to the desired choice of normalization.