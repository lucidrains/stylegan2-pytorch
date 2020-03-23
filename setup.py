from setuptools import setup, find_packages

setup(
  name = 'stylegan2_pytorch',
  packages = find_packages(),
  scripts=['bin/stylegan2_pytorch'],
  version = '0.4.16',
  license='GPLv3+',
  description = 'StyleGan2 in Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/stylegan2-pytorch',
  keywords = ['generative adversarial networks', 'artificial intelligence'],
  install_requires=[
      'fire',
      'numpy',
      'tqdm',
      'torch',
      'torchvision',
      'pillow',
      'torch_optimizer'
  ],
  classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3.6',
  ],
)