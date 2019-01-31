# Reaction Plane Fit Changelog

Changelog based on the [format here](https://keepachangelog.com/en/1.0.0/).

## [1.2] - 30 January 2019

### Added

- Added `mypy` to Travis CI checks. See: `3fe44b42773b7b6d4fa184cf8db82a5e529713e0`.
- Pre-commit hooks configuration to the repository based on `pre-commit`. This should generally improve the
  quality of commits. Currently includes `flake8`, `mypy`, `yamllint`, and some additional minor python
  checks. See: `e0218dffc0829ac742790497b5f0494bd41c0035` and `a40c72e42cb50bf0658c871e8ea8d3f0aa5b393e`.
- Ability to run Minos errors. They aren't stored (because it doesn't make sense to do so), but they can be
  compared to Hesse errors. See: `b0149099bc4e05a0e4f37d78680abe5ed4fa7ff3`.
- Allow for user arguments to override the fit arguments. See: `bf570c37163b57c02d890aa5047f99e8bb82077c`.

### Changed

- Adapted to new pachyderm histogram API. See: `c6b17fe6555d7baab3be9cb53c4f238bd8e23b65`.
- To implement the new API, it required some relaxation of array comparison tolerances (but they are still
  quite stringent). See: `631db84754b1d03076acc24ea5ab9bdae88bcb38`.
- Renamed "inPlane" -> "in_plane" (and the other angles accordingly) for better matching of conventions (which
  makes it easier to use and integrate with other packages). See: `300258f77f2ebd9f83754ab1d0f41e028e95032f`.
- Updated RP plot labeling for new RP orientation naming convention. See:
  `7a9e939f442a73e96349583553228d923341109a`.

### Fixed

- A number of type annotation issues identified by `mypy`. See: `8b71c77f99f8d95fd364bc7a700e2a32ecbdb499`.
- Test discovery failed in some cases. See: `85e5ad0f2a6ab4d55393167c9289018061826f37`

## [1.1] - 11 December 2018

### Added

- Test the `FitComponentResult` objects in the integration tests. See: `dcb21966`.
- Updated documentation.
- `mypy` to the dev requirements. Although not run in the CI (because not all of the identified issues are
  actually issues), it is good to have available and run from time to time. See: `bdc216a7`.

### Changed

- Refactored `Histogram` class to pachyderm, where it was renamed `Histogram1D`. See: `923c3847`.

## [1.0.1] - 30 November 2018

### Fixed

- Resolved a tagging problem related to a rebase.

## [1.0] - 30 November 2018

### Added

- Zenodo DOI information.
- Chi2/ndf calculation.
- Plot module (with tests) to draw the fit and the residual.
    - The tests take advantage of image comparion.
- Additional fit quality checks.

### Fixed

- Parameter argument ordering in the signal inclusive fit. See ba4e833a.
- Fixed error propagation by calculating the error for each individual component rather than for the
  simultaneous fit (which was the wrong approach). This has the side effect of substantially speeding up the
  error calculation.
- Specify all initial parameters, limits, and steps, which can substantially improve the chi2.
- Various type annotations.

## [0.9] - 27 November 2018

- This is essentially a preview release.
- It can perform the RP fit using background only data, including the reaction plane orientation inclusive
  signal, or the reaction plane orientation dependent signal. Full error propagation is performed for each
  fit component.

