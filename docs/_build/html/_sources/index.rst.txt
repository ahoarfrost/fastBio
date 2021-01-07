Welcome to fastBio's documentation!
===================================

fastBio is a library for manipulating data and creating and training deep learning models for biological sequencing data. It is an extension of the fastai v1 library.

A number of pretrained models for biological sequencing data can be loaded directly through fastBio with the **LookingGlass** and **LookingGlassClassifier** classes. 
These models are available for download at the sister repository `LookingGlass <https://github.com/ahoarfrost/LookingGlass>`_.

If you find fastBio or LookingGlass useful, please cite the preprint: 

      Hoarfrost, A., Aptekmann, A., Farfanuk, G. & Bromberg, Y. Shedding Light on Microbial Dark Matter with A Universal Language of Life. *bioRxiv* (2020). doi:10.1101/2020.12.23.424215. https://www.biorxiv.org/content/10.1101/2020.12.23.424215v2


.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. contents:: Table of Contents
    :depth: 3


Installation
==================
You can install fastBio with pip (python 3 only): ::

   pip3 install fastBio


Tutorial
==================

See the fastBio tutorial in `html <https://github.com/ahoarfrost/fastBio/blob/master/Tutorial.html>`_ or `jupyter notebook <https://github.com/ahoarfrost/fastBio/blob/master/Tutorial.ipynb>`_ form.


API Docs
==================

Transforms
------------------
BioTokenizer
+++++++++++++++++++
   .. autoclass:: transform.BioTokenizer
       :members:

BioVocab
+++++++++++++++++++
   .. autoclass:: transform.BioVocab
       :members:


Databunching
------------------
BioDataBunch
+++++++++++++++++++
   .. autoclass:: data.BioDataBunch
       :members:

BioLMDataBunch
+++++++++++++++++++
   .. autoclass:: data.BioLMDataBunch

BioClasDataBunch
+++++++++++++++++++
   .. autoclass:: data.BioClasDataBunch


Models
------------------

LookingGlass
+++++++++++++++++++
   .. autoclass:: models.LookingGlass
      :members:

LookingGlassClassifier
++++++++++++++++++++++++++++++++++++++
   .. autoclass:: models.LookingGlassClassifier
      :members:


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
