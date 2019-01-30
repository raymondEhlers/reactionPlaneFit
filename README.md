# Reaction plane fit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1599239.svg)](https://doi.org/10.5281/zenodo.1599239)
[![Documentation Status](https://readthedocs.org/projects/reactionplanefit/badge/?version=latest)](https://reactionplanefit.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.com/raymondEhlers/reactionPlaneFit.svg?branch=master)](https://travis-ci.com/raymondEhlers/reactionPlaneFit)
[![codecov](https://codecov.io/gh/raymondEhlers/reactionPlaneFit/branch/master/graph/badge.svg)](https://codecov.io/gh/raymondEhlers/reactionPlaneFit)

Implements the reaction plane (RP) fit described in [Phys. Rev. C 93,
044915](https://journals.aps.org/prc/abstract/10.1103/PhysRevC.93.044915) (or on the
[arxiv](https://arxiv.org/abs/1509.04732)) to characterize and subtract background contributions to
correlation functions measured in heavy ion collisions. It allows for fits to the background dominated region,
as well as the option of fitting the RP inclusive signal dominated region or the RP dependent signal dominated
region. This package implements the fit for three orientations relative to the RP, with the possibility of
straightforward extension to other sets of orientations.

![Sample reaction plane fit](https://github.com/raymondEhlers/reactionPlaneFit/raw/master/docs/images/sampleSignalInclusiveRPF.png)

## Installation

This package requires python 3.6 and above. A few prerequisites are required which unfortunately cannot be
resolved solely by pip because of the packaging details of `probfit`.

```bash
$ pip install numpy cython
```

The package is available on [PyPI](https://pypi.org/project/reaction_plane_fit) and is available via pip.

```bash
$ pip install reaction_plane_fit
```

You may want to consider installing this to your user directory (`--user`) or better, within a virtual
environment.

# Usage

Performing a fit with this package only requires a few lines of code. Below is sufficient to define and run a
fit with some sample values:

```python
from reaction_plane_fit import three_orientations
# Define the fit object.
rp_fit = three_orientations.BackgroundFit(
    resolution_parameters = {"R22": 0.6, "R42": 0.3, "R62": 0.1, "R82": 0.1},
    use_log_likelihood = False,
    signal_region = (0, 0.6),
    background_region = (0.8, 1.2),
)
# Load or otherwise provide the relevant histograms here.
# The structure of this dictionary is important to ensure that the proper data ends up in the right place.
# Note that the data must already be projected into the background or signal dominated regions.
data = {"background": {"in_plane": ROOT.TH1.., "mid_plane": ROOT.TH1..., "out_of_plane": ROOT.TH1...}}
# Perform the actual fit.
success, _ = rp_fit.fit(data = data)
# Print the fit results
print("Fit result: {fit_result}".format(fit_result = rp_fit.fit_result))
```

Examples for fitting an inclusive reaction plane orientation signal alongside the background, or for fitting
only the background are both available in `reaction_plane_fit.example`. This module can also be run directly
in the terminal via:

```
$ python -m reaction_plane_fit.example [-b] [-i dataFilename]
```

If fit data is not specified, it will use some sample data. For further information, including all possible
fit function combinations, please see the [full documentation](https://reactionplanefit.readthedocs.io/en/latest/).

# Fits implemented

There are three possible types of fits:

- Background dominated region only: This fits only regions on the near-side at large dEta which are dominated
  by background contributions. Called `BackgroundFit` in each set of orientations implementation.
- Inclusive signal region: This fits to the RP inclusive signal dominated region, as well as fitting the RP
  dependent background dominated regions. Called `InclusiveSignalFit` in each set of orientations implementation.
- RP dependent signal region: This fits to the RP dependent signal dominated region, as well as fitting the RP
  dependent background dominated regions. Called `SignalFit` in each set of orientations implementation.

## Three orientation (in-, mid-, and out-of-plane)

This package implements the fit for three orientations relative to the reaction plane:

- in-plane (0<|&Delta;&phi;|<&pi;/6)
- mid-plane (&pi;/6<|&Delta;&phi;|<&pi;/3)
- and out-of-plane (&pi;/3<|&Delta;&phi;|<&pi;/2)

These fits are implemented in the `reaction_plane_fit.three_orientations` module.

# Development

If developing the packaging, clone the repository and then install with

```bash
$ pip install -e .[dev,tests]
```

Note that python 3.6 and above is required because this package uses `dataclasses` (which has a python 3.6
backport), and it relies on dictionaries being ordered (which is true for `cpython` 3.6 and is required for
python 3.7 in general).

# Citation

Please cite the paper (Phys. Rev. C 93, 044915), as well as this implementation:

```
@misc{raymond_ehlers_2018_1599239,
  author       = {Raymond Ehlers},
  title        = {reactionPlaneFit - RPF implementation},
  month        = nov,
  year         = 2018,
  doi          = {10.5281/zenodo.1599239},
  url          = {https://doi.org/10.5281/zenodo.1599239}
}
```

# Acknowledgments

Code started from implementation work done by [M. Arratia](https://github.com/miguelignacio/BackgroundFit).
Thanks to C. Nattrass and J. Mazer for help and discussions.
