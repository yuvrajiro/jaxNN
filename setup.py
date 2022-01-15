from setuptools import setup

setup(name='jaxNN',
    version='0.1',
    description='jaxNN',
    author='Rahul Goswami',
    author_email='yuvrajiro@gmail.com',
    license='MIT',
    packages=['jaxNN'],
    install_requires=['jax', 'jaxlib', 'numpy', 'scipy', 'tqdm'],
    scripts=['test/test_jaxNN.py'],
    zip_safe=False)