from setuptools import setup, find_packages

setup(
  name = 'stylegan2_pytorch',
  packages = find_packages(),
  entry_points={
      'console_scripts': [
          'stylegan2_pytorch = stylegan2_pytorch.cli:main',
      ],
  },
  version = '1.1.5',
  license='GPLv3+',
  description = 'StyleGan2 in Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/stylegan2-pytorch',
  download_url = 'https://github.com/lucidrains/stylegan2-pytorch/archive/v_036.tar.gz',
  keywords = ['generative adversarial networks', 'artificial intelligence'],
  install_requires=[
      'contrastive_learner>=0.1.0',
      'fire',
      'kornia',
      'linear_attention_transformer',
      'numpy',
      'retry',
      'tqdm',
      'torch',
      'torchvision',
      'pillow',
      'pytorch-fid',
      'vector-quantize-pytorch'
  ],
  classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3.6',
  ],
)