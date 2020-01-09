# stylegan2-pytorch

Simple working implementation of Stylegan2 in Pytorch

## Requirements

You will need a machine with a GPU and CUDA installed

```
python3.6
pytorch
numpy
```

```
> pip install -r requirements.txt
```

## Usage

```
> python train.py --data /path/to/images
```

Sample images will be saved to `results/default` and models will be saved periodically to `models/default`

You can specify the name of your project with

```
> python train.py --data /path/to/images --name my-project-name
```

By default, if the training gets cut off, it will automatically resume from the last checkpointed file. If you want to restart with new settings, just add a `new` flag

```
> python train.py --new --data /path/to/images --name my-project-name --image-size 512
```

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