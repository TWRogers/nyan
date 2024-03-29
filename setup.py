from setuptools import find_packages, setup
from nyan.version import __version__


def readme():
    with open('README.md') as f:
        return f.read()


extras_require = {}

setup(name='nyan',
      author='Thomas W. Rogers',
      author_email='thomas.rogers08@gmail.com',
      version=__version__,
      description='A library for image and video transformation and mapping.',
      long_description=readme(),
      long_description_content_type='text/markdown',
      url='https://github.com/TWRogers/nyan',
      license='Apache 2.0',
      packages=find_packages(),
      include_package_data=True,
      python_requires='>=3.5.2',
      setup_requires=[
          'pytest-runner'
      ],
      install_requires=[
          'Pillow',
          'numpy',
          'imageio',
          'opencv-python',
          'imageio-ffmpeg',
          'matplotlib',
          'scipy'
      ],
      tests_require=[
          'pytest',
          'flake8'
      ],
      extras_require=extras_require,
      test_suite='tests',
      zip_safe=False)
