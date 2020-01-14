# Simple StyleGan2 for Pytorch

<img src="https://raw.githubusercontent.com/lucidrains/stylegan2-pytorch/master/sample.jpg" width="350" height="350">

Simple working Pytorch implementation of Stylegan2 based on https://arxiv.org/abs/1912.04958

## Install

You will need a machine with a GPU and CUDA installed. Then pip install the package like so

```
> pip install stylegan2_pytorch
```

## Use

```
> stylegan2_pytorch --data /path/to/images
```

That's it. Sample images will be saved to `results/default` and models will be saved periodically to `models/default`

## Advanced Use

You can specify the name of your project with

```
> stylegan2_pytorch --data /path/to/images --name my-project-name
```

By default, if the training gets cut off, it will automatically resume from the last checkpointed file. If you want to restart with new settings, just add a `new` flag

```
> stylegan2_pytorch --new --data /path/to/images --name my-project-name --image-size 512 --network-capacity 20
```

## Todo

1. Add mixed precision and multi-GPU support

## Appreciation

Thank you to Matthew Mann for his inspiring [simple port](https://github.com/manicman1999/StyleGAN2-Tensorflow-2.0) for Tensorflow 2.0

## References

```
@article{Karras2019stylegan2,
  title   = {Analyzing and Improving the Image Quality of {StyleGAN}},
  author  = {Tero Karras and Samuli Laine and Miika Aittala and Janne Hellsten and Jaakko Lehtinen and Timo Aila},
  journal = {CoRR},
  volume  = {abs/1912.04958},
  year    = {2019},
}
```
