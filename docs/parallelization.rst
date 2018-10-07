Parallelization
=========

Most of the heavy computation within diffxpy functions is carried out by batchglm. batchglm uses numpy and tensorflow for the run-time limiting linear algebra operations. Both tensorflow and numpy may show different parallelization behaviour depending on the operating system. Here, we describe how one can limit the number of cores used by diffxpy by controlling its dependencies, numpy and tensorflow. Note that these limits may not be necessary on all platforms. Secondly, also note that such limits lead to suboptimal performance given the total resources of your machine.

tensorflow
----------

We propagated direct control of tensorflow multi-threading into batchglm which can be called as follows:

    from batchglm.pkg_constants import TF_CONFIG_PROTO
    TF_CONFIG_PROTO.inter_op_parallelism_threads = 1
    TF_CONFIG_PROTO.intra_op_parallelism_threads = 1

Note that tensorflow distinguishes intra- and inter-operation parallelism which refers to how tasks are distributed across threads. To avoid this strict allocation you can also set the following environmental variable that is used by tensorflow for the entire pool of threads:

    import os
    os.environ.setdefault("TF_NUM_THREADS", "1")

numpy/scipy
-----------

Numpy/scipy multi-threading in the linalg sub-modules can be controlled as follows in the shell in which the python session is started in which diffxpy is used (e.g. the shell from which jupyter notebook is called):

    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export OMP_NUM_THREADS=1

Here, we restricted the number of threads to be used by numpy to 1. Numpy is not used for the run-time determining parameter estimation steps so that a larger number of threads has little effect on the overall run time. So far, we have only observed this to be necessary on linux operating systems.