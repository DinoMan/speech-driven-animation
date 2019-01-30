from setuptools import setup

setup(name='bulletin',
      version='0.1',
      description='helper library for posting to webserver',
      packages=['bulletin'],
      package_dir={'bulletin': 'bulletin'},
      install_requires=[
          'matplotlib',
          'menpo',
          'scipy',
          'numpy'
      ],
      zip_safe=False)

