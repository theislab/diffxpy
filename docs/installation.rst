Installation
============

We assume that you have a python environment set up.

Firstly, you need to install batchglm which depdends on the PyPi packages and tensorflow and tensorflow-probability.
You can install these dependencies from source to optimze them to your hardware which can improve performance.
Note that both packages also have GPU versions which allows you to run the run-time limiting steps of diffxpy on GPUs.
The simplest installation of these dependencies is via pip: call::

    pip install tf-nightly
    pip install tfp-nightly

The nightly versions of tensorflow and tensorflow-probability are up-to-date versions of these packages.
Alternatively you can also install the major releases: call::

    pip install tensorflow
    pip install tensorflow-probability


You can then install batchglm from source by using the repository on `GitHub
<https://github.com/theislab/batchglm>`__: 

- Chose a directory where you want batchglm to be located and ``cd`` into it.
- Clone the batchglm repository into this directory.
- ``cd`` into the root directory of batchglm.
- Install batchglm from source: call::

    pip install -e .

Finally, you can then install diffxpy from source by using the repository on `GitHub
<https://github.com/theislab/diffxpy>`__: 

- Chose a directory where you want batchglm to be located and ``cd`` into it.
- Clone the diffxpy repository into this directory.
- ``cd`` into the root directory of diffxpy.
- Install diffxpy from source: call::

    pip install -e .

You can now use diffxpy in a python session by via the following import: call::

    import diffxpy.api as de 
