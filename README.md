# Reaction plane fit

Implements the reaction plane fit described in [Phys. Rev. C 93,
044915](https://journals.aps.org/prc/abstract/10.1103/PhysRevC.93.044915)
([arxiv](https://arxiv.org/abs/1509.04732)) to characterize and subtract background contributions to
correlation functions measured in heavy ion collisions. This package implements the fit for 3 angles relative
to the reaction plane, in-plane ($0<|\Delta\varphi|<\pi/6$), mid-plane ($\pi/6<|\Delta\varphi|<\pi/3$), and
out-of-plane ($\pi/3<|\Delta\varphi|<\pi/2$).

![Sample reaction plane fit](.github/sample.png)

## Installation

A few prerequisites are required which unfortunately cannot be resolved solely by pip because of problems with
pip and because some other packages (`iminiut` and `probfit`) do not have recent enough tags.

```{bash}
$ pip install numpy cython
# Must be on a separate line from the above!
$ pip install git+https://github.com/raymondEhlers/iminuit.git@master#egg=iminuit git+https://github.com/raymondEhlers/probfit.git@master#egg=probfit
```

The package is available on [pypi](https://pypi.org/project/reactionPlaneFit) and is available via pip.

```{bash}
$ pip install --user reactionPlaneFit
```

This assumes installation for only the current user. 

## Usage

To perform the fit, a number of parameters need to be specified. In particular, the reaction plane resolution
parameters must be specified. An example configuration is in the `config/` folder. Example data is in the
`data/` folder.

# Citations

Cite the paper, this implementation, 

# Development

If developing the packaging, clone the repository and then install with

```{bash}
$ pip install -e . --process-dependency-links
```

`pip` will warn that processing dependency links is deprecated - you can safely ignore this. It is required to
properly resolve the dependencies and is not actually deprecated. For tests, don't forget to install `pytest`!

# Acknowledgments

Code started from implementation work done by [M. Arratia](https://github.com/miguelignacio/BackgroundFit).
Thanks to C. Nattrass and J. Mazer for help and discussions.
