## Simple StyleGan2 for Pytorch
[![PyPI version](https://badge.fury.io/py/stylegan2-pytorch.svg)](https://badge.fury.io/py/stylegan2-pytorch)

Simple working Pytorch implementation of Stylegan2 based on https://arxiv.org/abs/1912.04958

Below are some flowers that do not exist.

<img src="https://raw.githubusercontent.com/lucidrains/stylegan2-pytorch/master/samples/flowers.jpg" width="450" height="auto">

Neither do these hands

<img src="https://raw.githubusercontent.com/lucidrains/stylegan2-pytorch/master/samples/hands.jpg" width="450" height="auto">

## Install

You will need a machine with a GPU and CUDA installed. Then pip install the package like this

```bash
$ pip install stylegan2_pytorch
```

If you are using a windows machine, the following commands reportedly works.

```bash
$ conda install pytorch torchvision -c python
$ pip install stylegan2_pytorch
```

## Use

```bash
$ stylegan2_pytorch --data /path/to/images
```

That's it. Sample images will be saved to `results/default` and models will be saved periodically to `models/default`.

## Advanced Use

You can specify the name of your project with

```bash
$ stylegan2_pytorch --data /path/to/images --name my-project-name
```

You can also specify the location where intermediate results and model checkpoints should be stored with

```bash
$ stylegan2_pytorch --data /path/to/images --name my-project-name --results_dir /path/to/results/dir --models_dir /path/to/models/dir
```

By default, if the training gets cut off, it will automatically resume from the last checkpointed file. If you want to restart with new settings, just add a `new` flag

```bash
$ stylegan2_pytorch --new --data /path/to/images --name my-project-name --image-size 512 --batch-size 1 --gradient-accumulate-every 16 --network-capacity 10
```

Once you have finished training, you can generate images from your latest checkpoint like so.

```bash
$ stylegan2_pytorch  --generate
```

If a previous checkpoint contained a better generator, (which often happens as generators start degrading towards the end of training), you can load from a previous checkpoint with another flag

```bash
$ stylegan2_pytorch --generate --load-from {checkpoint number}
```

## Memory considerations

The more GPU memory you have, the bigger and better the image generation will be. Nvidia recommended having up to 16GB for training 1024x1024 images. If you have less than that, there are a couple settings you can play with so that the model fits.

```bash
$ stylegan2_pytorch --data /path/to/data \
    --batch-size 3 \
    --gradient-accumulate-every 5 \
    --network-capacity 16
```

1. Batch size - You can decrease the `batch-size` down to 1, but you should increase the `gradient-accumulate-every` correspondingly so that the mini-batch the network sees is not too small. This may be confusing to a layperson, so I'll think about how I would automate the choice of `gradient-accumulate-every` going forward.

2. Network capacity - You can decrease the neural network capacity to lessen the memory requirements. Just be aware that this has been shown to degrade generation performance.

## Todo

1. Add mixed precision and multi-GPU support

## Appreciation

Thank you to Matthew Mann for his inspiring [simple port](https://github.com/manicman1999/StyleGAN2-Tensorflow-2.0) for Tensorflow 2.0

This also uses the `hinge L∞ gradient penalty` described in https://arxiv.org/abs/1910.06922

## References

```bibtex
@article{Karras2019stylegan2,
  title   = {Analyzing and Improving the Image Quality of {StyleGAN}},
  author  = {Tero Karras and Samuli Laine and Miika Aittala and Janne Hellsten and Jaakko Lehtinen and Timo Aila},
  journal = {CoRR},
  volume  = {abs/1912.04958},
  year    = {2019},
}
```

```bibtex
@article{jolicoeur2019connections}
  title={Connections between Support Vector Machines, Wasserstein distance and gradient-penalty GANs},
  author={Jolicoeur-Martineau, Alexia},
  journal={arXiv preprint arXiv:1910.06922},
  year={2019}
}
```

```bibtex
@article{,
  title= {Oxford 102 Flowers},
  author= {Nilsback, M-E. and Zisserman, A., 2008},
  abstract= {A 102 category dataset consisting of 102 flower categories, commonly occuring in the United Kingdom. Each class consists of 40 to 258 images. The images have large scale, pose and light variations.}
}
```

```bibtex
@article{afifi201911k,
  title={11K Hands: gender recognition and biometric identification using a large dataset of hand images},
  author={Afifi, Mahmoud},
  journal={Multimedia Tools and Applications},
  volume={78},
  number={15},
  pages={20835--20854},
  year={2019},
  publisher={Springer}
}
```
