from setuptools import setup

setup(name='sda',
      version='0.2',
      description='Produces speech-driven faces',
      packages=['sda'],
      package_dir={'sda': 'sda'},
      package_data={'sda': ['data/*.dat']},
      install_requires=[
          'numpy',
          'scipy',
          'scikit-video',
          'scikit-image',
          'ffmpeg-python',
          'torch',
          'face-alignment',
          'torchvision',
          'pydub',
          ],
      zip_safe=False)
