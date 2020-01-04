itertools-len
=============

|pkg| |docs| |ci| |cov|

.. |pkg| image:: https://img.shields.io/pypi/v/itertools-len
   :target: https://pypi.org/project/itertools-len
.. |docs| image:: https://readthedocs.org/projects/itertools-len/badge/?version=latest
   :target: https://itertools-len.readthedocs.io
.. |ci| image:: https://travis-ci.com/flying-sheep/itertools-len.svg?branch=master
   :target: https://travis-ci.com/flying-sheep/itertools-len
.. |cov| image:: https://codecov.io/gh/flying-sheep/itertools-len/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/flying-sheep/itertools-len

Have you ever been annoyed that the length information of ``itertools`` have not been preserved?

This module faithfully wraps every one of them (together with ``map``) where ``len`` can be derived:

>>> from itertools_len import chain, product
>>> len(chain('abc', [1, 2]))
5
>>> len(product('abc', [1, 2]))
6
