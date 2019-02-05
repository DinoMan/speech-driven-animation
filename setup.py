from setuptools import setup

setup(name='sda',
      version='0.1',
      description='Produces speech-driven faces',
      packages=['sda'],
      package_dir={'sda': 'sda'},
      package_data={'sda': ['data/*.npy']},
      install_requires=[
          'face_alignment',
          'numpy',
          'scipy'
      ],
      zip_safe=False)