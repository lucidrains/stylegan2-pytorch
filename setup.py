import sys
from setuptools import setup, find_packages

sys.path[0:0] = ['stylegan2_pytorch']
from version import __version__

setup(
  name = 'stylegan2_pytorch',
  packages = find_packages(),
  entry_points={
      'console_scripts': [
          'stylegan2_pytorch = stylegan2_pytorch.cli:main',
      ],
  },
  version = __version__,
  license='GPLv3+',
  description = 'StyleGan2 in Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/stylegan2-pytorch',
  download_url = 'https://github.com/lucidrains/stylegan2-pytorch/archive/v_036.tar.gz',
  keywords = ['generative adversarial networks', 'artificial intelligence'],
  install_requires=[
      'aim',
      'einops',
      'contrastive_learner>=0.1.0',
      'fire',
      'kornia>=0.5.4',
      'numpy',
      'retry',
      'tqdm',
      'torch',
      'torchvision',
      'pillow',
      'vector-quantize-pytorch>=0.1.0'
  ],
  classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3.6',
  ],
)