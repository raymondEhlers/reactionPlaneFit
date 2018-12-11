# Reaction Plane Fit Changelog

Changelog based on the [format here](https://keepachangelog.com/en/1.0.0/).

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

