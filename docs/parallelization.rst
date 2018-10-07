Parallization
=============

Most of the heavy computation within diffxpy functions is carried out by batchglm. batchglm uses numpy and tensorflow for the run-time limiting linear algebra operations. Both tensorflow and numpy may show different parallelization behaviour depending on the operating system. Here, we describe how one can limit the number of cores used by diffxpy by controlling its dependencies, numpy and tensorflow. Note that these limits may not be necessary on all platforms. Secondly, also note that such limits lead to suboptimal performance given the total resources of your machine.

tensorflow
----------

Tensorflow multi-threading can be set before batchglm (and therefore diffxpy) are imported into a python session. Accordingly, you have to restart your python session if you want to change the current parallelization settings. Parallelization of tensorflow can be controlled via the following two environmental variables: call::

    # Before importing diffxpy.api or batchglm.api in your python session, execute:
    import os
    os.environ.setdefault("TF_NUM_THREADS", "1")
    os.environ.setdefault("TF_LOOP_PARALLEL_ITERATIONS", "1")
    import diffxpy.api as de

TF_NUM_THREADS controls the number of threads that are used for linear algebra operations in tensorflow, which controls parallelization during training. TF_LOOP_PARALLEL_ITERATIONS controls the number of threads which are used during tensorflow while_loops, which are used during hessian computation. Here, we set both to one so that only one thread is used by tensorflow within diffxpy.

The environmental variables are checked upon loading of batchglm and are converted into package constants which control the parallelization behaviour of tensorflow. These package constants can also be set after package loading, but they do not affect the behaviour anymore once a tensorflow session was started once. If you want to set parallilzation behaviour after loading the package but before fist using it, you can therefore run: call::

    import diffxpy.api as de
    from batchglm.pkg_constants import TF_CONFIG_PROTO
    TF_CONFIG_PROTO.inter_op_parallelism_threads = 1
    TF_CONFIG_PROTO.intra_op_parallelism_threads = x
    from batchglm.pkg_constants import TF_LOOP_PARALLEL_ITERATIONS
    TF_LOOP_PARALLEL_ITERATIONS = x

where x is the number of threads (integer) to be used within diffxpy.


numpy/scipy
-----------

Numpy/scipy multi-threading in the linalg sub-modules can be controlled as follows in the shell in which the python session is started in which diffxpy is used (e.g. the shell from which jupyter notebook is called): call::

    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export OMP_NUM_THREADS=1

Here, we restricted the number of threads to be used by numpy to 1. Numpy is not used for the run-time determining parameter estimation steps so that a larger number of threads has little effect on the overall run time. So far, we have only observed this to be necessary on some linux operating systems.