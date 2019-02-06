from setuptools import setup

setup(name='sda',
      version='0.1',
      description='Produces speech-driven faces',
      packages=['sda'],
      package_dir={'sda': 'sda'},
      package_data={'sda': ['data/*.dat']},
      install_requires=[
          'numpy',
          'scipy',
          'scikit-video',
          'ffmpeg-python',
      ],
      zip_safe=False)