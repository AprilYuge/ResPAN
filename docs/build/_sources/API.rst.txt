API document
====================

This section provides detailed API documentation for all public functions
and classes in ``ResPAN``.

Preprocessing
-----------------

.. code-block:: python

   def data_preprocessing(adata):
      """Function used to preprocess our data with batch effect
      """

- **adata**: The datasets with batch effect in AnnData form.

ResPAN
-----------------

.. code-block:: python
   
   class Mish(nn.Module):
    """A class implementation for the state-of-the-art activation function, Mish.
    """

.. code-block:: python

   class discriminator(nn.Module):
    """The discrimimator structure of our AWGAN. 
        Layer: N(2000)->1024->512->256->128->1
    """
.. code-block:: python

   class generator(nn.Module):
      """The generator structure of our AWGAN. 
      Layer: 2000->1024->512->256->512->1024->2000
      with skip connection
      """

.. code-block:: python

   def run_respan(adata, batch_key='batch', order=None, epoch=300, batch=1024, lambda_1=10., 
               reduction='pca', subsample=3000, k1=None, k2=None, filtering=False,
               n_critic=10, seed=999, b1=0.9, b2=0.999, lr=0.0001, opt='AdamW'):
   """The main entry of our model
   """

- **adata**: Data with batch effect.
- **batch_key**:  The index name of batch information in the given dataset.
- **order**: The batch sequence for training or none.
- **epoch**: The number of iteraiton steps.
- **batch**: Input batch.
- **lambda_1**: A hyperparameter used to control the weights of gradient penalty.
- **reduction**: Method we used for dimension reduction, including None, 'cca', 'pca' and 'kpca'. 
- **subsample**: The size of our data used to generate training dataset.
- **k1**: The number of nearest neighbors we used across different batches.
- **k2**: The number of nearest neighbors we used inner one batch.
- **filtering**: A bool value used to determine whether we used top features selection or not.
- **n_critic**: A step value used to determine the training times of generator. 
- **seed**: Random seed we used in our training process.
- **b1,b2**: Hyperparameters used in AdamW optimizer.
- **lr**: Learning rate.
- **opt**: The name of our optimizer.