itertools-len
=============

Have you ever been annoyed that the length information of ``itertools`` have not been preserved?

This module faithfully wraps every one of them (together with ``map``) where ``len`` can be derived:

>>> from itertools_len import chain, product
>>> len(chain('abc', [1, 2]))
5
>>> len(product('abc', [1, 2]))
6
