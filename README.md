### Hyperparameter Optimization made Simple

This is a wrapper of James Bergstra's [hyperopt-convnet](https://github.com/hyperopt/hyperopt-convnet) aimed not only at simplification but also at compliance with Nicolas Pinto's [family](http://pinto.scripts.mit.edu/uploads/Research/fg2011_lfw_final_v2.pdf) of biologically-inspired visual features, i.e., convolutional networks (convnets).

The code has been used by our research group, but it has not been extensively tested. Therefore, one should expect some bumps while using it. We appreciate if you report them. To use the package *as is*, go with

```
python setup.py install
```

and refer to the scripts in `simplehp/scripts` to:

- `hp-dataset.py` : run hyperparameter optimization;
- `check-hp.py` : check optimization status;
- `protocol-eval.py`: evaluate best found convnet according to the dataset's protocol.

If you want to use the package to optimize architectures on your problems of interest, then you'd better go with

```
python setup.py develop
```

and code your own *data provider*. This package includes an example of data provider for the [PubFig83](https://www.dropbox.com/s/0ez5p9bpjxobrfv/pubfig83-aligned.tar.bz2) and [CalTech256](http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar) datasets in `simplehp/data/rndsplits.py`. You may want to use it as a starting point.

Indeed, the possibility of coding a data provider *short and sweet* is of great appeal in `simple-hp`. Have a look at `simplehp/data/base.py` to find this out.

Finally, in `extra` you will find installation scripts for installation of the whole stack of libraries required to use `simple-hp`. They were used in Ubuntu 12.04, and have been poorly tested. Take this as a help to get you on track. No more than that.
