soliton — tensors, autograd and ops
===================================

.. currentmodule:: soliton

Everything here is importable straight from ``soliton``.

Devices and memory
------------------

.. autofunction:: soliton.tensor.plan
.. autofunction:: soliton.tensor.memory_stats
.. autofunction:: soliton.tensor.set_memory_limit
.. autofunction:: soliton.tensor.reset_peak_stats
.. autofunction:: soliton.tensor.empty_cache
.. autofunction:: soliton.tensor.cuda_mem_info
.. autofunction:: soliton.tensor.set_device
.. autofunction:: soliton.tensor.set_tf32
.. autofunction:: soliton.tensor.set_fusion
.. autofunction:: soliton.tensor.synchronize

The tensor
----------

.. autoclass:: soliton.tensor.Tensor
   :members:

Creation
--------

.. autofunction:: soliton.tensor.tensor
.. autofunction:: soliton.tensor.empty
.. autofunction:: soliton.tensor.zeros
.. autofunction:: soliton.tensor.full
.. autofunction:: soliton.tensor.normal

Operations
----------

.. autofunction:: soliton.tensor.add
.. autofunction:: soliton.tensor.sub
.. autofunction:: soliton.tensor.mul
.. autofunction:: soliton.tensor.div
.. autofunction:: soliton.tensor.scale
.. autofunction:: soliton.tensor.matmul
.. autofunction:: soliton.tensor.linear
.. autofunction:: soliton.tensor.gelu
.. autofunction:: soliton.tensor.softmax
.. autofunction:: soliton.tensor.layernorm
.. autofunction:: soliton.tensor.attention
.. autofunction:: soliton.tensor.embedding
.. autofunction:: soliton.tensor.cross_entropy
.. autofunction:: soliton.tensor.reshape
.. autofunction:: soliton.tensor.permute
.. autofunction:: soliton.tensor.transpose
.. autofunction:: soliton.tensor.sum_
.. autofunction:: soliton.tensor.clone
.. autofunction:: soliton.tensor.sumsq

Autograd
--------

.. autofunction:: soliton.tensor.backward
.. autofunction:: soliton.tensor.no_grad
.. autofunction:: soliton.tensor.checkpoint
