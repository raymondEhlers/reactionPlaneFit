# Reaction Plane Fit Changelog

Changelog based on the [format here](https://keepachangelog.com/en/1.0.0/).

## [3.1] - 24 August 2019

## Changed

- Added a method to create a full set of fit components for each EP orientation, regardless of which were used
  for the fit. This allows for downstream code to not need to differentiate between the different cases for
  most tasks, easing the user experience. See: Nearly all of the commits between 3.0 and 3.1.
- Use the `pachyderm.fit.BaseFitResult` for components in order to avoid needing to calculate unnecessary
  values for component fit results. It's more natural. See: `bbbcac45c41490072f345b6aebbeb6f49d427a1d`.
- Updated the included test data for better test consistently with external sources. See:
  `0bf5e1ceaede9cd578a20c12db81455f38326a2d`.
- Updated pre-commit hooks. See: `6d780545b5547f42c7c3a684f11234d069973c10`.

## Fixed

- Warning about registering `pytest` marks. See: `8f6f5386fddea1e540a15de99ce8852dfc2319fd`.
- Removed final `probfit` dependence. See: `b94ea052246269e454f3df1c7afbb7ac96f35cec`.

## [3.0] - 8 August 2019

## Changed

- Move to using the pachyderm fit package for general fit functionality, cost functions, and fit results. This
  move was instigated by the recognition that `probfit` cost functions didn't actually work properly with
  pre-binned data. It also allows for the centralization of basic fit code to make it generally available.
  Some code from this package was refactored into pachyderm. See: `9e62f6777afed290709b74d4ee18284d61c2bab9`
  and `0f66cff8264ff06f5138d78ba0213f40a195dd5e`. Note that developed was handled in a private branch in this
  repository, so some older commits have references to a fit package in the RPF package before it was moved to
  pachyderm.
- No longer rely on `probfit`. All functionality and more is now provided in `pachyderm.fit`. Plus, the package
  wasn't super actively maintained. See: `9e62f6777afed290709b74d4ee18284d61c2bab9`.
- Updated pre-commit hooks. See: `03844495a12f4269b14dbd4972c09fb203a88c7c`.

## Fixed

- Log likelihood and chi squared cost functions properly take binned data into account. See:
  `0f66cff8264ff06f5138d78ba0213f40a195dd5e`.

## [2.1.2] - 4 May 2019

### Added

- Enabled strict `mypy` checks. See: `1585c7398201319bac49f6881f678be3f9582ee5`.

### Changed

- Improved typing. `341330a5e48f1526848de27ceaba4ee10499de67`.

## [2.1.1] - 3 May 2019

### Changed

- Refactor function error calculation so that it can used by other packages. See:
  `04dff6cc7cbd1fd4ad1f2f26ecc528dfeebbe3f7`.

### Fixed

- Enable fixing parameters via user arguments. See: `cf9de684245217d4c7583407940d5a3864b192b1`.
- Properly mark the package as typed. See: `fe43d5368bb8fdd1defa5dbff8a39ac45d2424a3`.
- Improved code formatting and typing information, and fixed typos.
- Fix `mypy` typing in third party packages by adding `zip_safe = False` in the `setup.py`. See:
  `79eb4e821bed8d5290c18c779c2bce6c8a71e5c0`.

## [2.1] - 29 March 2019

### Added

- Calculate the correlation matrix in the fit result. See: `ee94cb80e774c38848f31592b8df765c333fba7a`.
- Read and write fit results with YAML, including integration tests. See:
  `983a2519d9b882a34d6bf44be312b45a3e051316`.
- Added test for invalid fit arguments. See: `009a974dc038d79ad2890e9c3f935094a88cb3ee`.

### Changed

- Modify the `ReactionPlaneFit.fit(...)` to return the minuit object for advanced use cases. The fit result
  stores almost all of the minuit information, but it can be useful in limited circumstances. For now, it is
  only used to check the correlation matrix calculation. See: `7a1efd908203abf1a1b8cb8d2f9d76afdfb1442d`.
- Initialize the component fit functions when defining the fit objects. This allows fit components to be
  evaluated using stored fit results. See: `e2673f25e7fa952087bb45ab5e35204850138b69`.
- Updated typing information.  See: `0f284c79ccfb6ee04715b33e4500b2c96832f7ff` and
  `60af85240d84adf5e011ee0b4b2f7a0c8cef154e`.
- Updated documentation. See: `b94fc955ce15034490888ec75368853f61ad9244`.
- Updated pre-commit hooks with newer versions of the packages. See:
  `6d54231e4707502fec9811c0507daa5a37358076`.

## [2.0] - 22 February 2019

Note the API changes introduced by moving the fit results to the components!

### Added

- Store errors on fit parameters. See: `fbb9bd36cf7f4b010377cb1c1d27cb5d09ccf57d`.
- Evaluate the background function of a component. For a background fit component, this is the same as the fit
  function, while for a signal fit component, this will describe only the background contributions. See:
  `8836a38cbdb44f6db1508a23f295ff4872433938`.
- Preliminary instructions on running the fit via c++. They are not yet tested. See:
  `8f6d57ca160e9500b4fa5c2747faa9e4b23c5a12`.

### Changed

- Moved component fit results into the fit components. See: `a08c42a483d51ba5c599986ae8f5cdec6a4505f0` and
  `51ef75007d2a765bd03bf01958517d3109140067`.
- Moved fit evaluation into the fit component. This was enabled by moving the fit results into the fit
  components. See: `e3c13956b80ad31b7232f46fe0784b0179462175`.
- Moved `FitType` to the `base` module so that it is more easily accessible. See:
  `f3513c9f3d01d51c46e6b776fad005c32864001d`.
- Moved error calculation to fit component. See: `674c8cf0f3bff1ebe95a8d5b76a7c2917d3169b7`.
- Refactor error calculation to allow it to calculate errors for both the fit and background functions (or any
  other function). See: `674c8cf0f3bff1ebe95a8d5b76a7c2917d3169b7`.
- Improved example plot styling. See: `eb9fe16bd19e1e0106a822ee7e6a31f8a9be7e51` (included with a few other
  minor changes).
- Determine and plot errors on residual. See: `74b7f14b34b37b3563dcebc1031730a9343b5252`.
- Large improvements to typing information and documentation in a variety of commits.

### Fixed

- Background fit example legend placement was accidentally moved. See:
  `1869eb0650c9b26157c6fed103e80e5ad78f7f08`.
- Baseline test images, which started failing after a number of small aesthetic changes. See:
  `795d9d228a5cde6d5477dfd1c697efcabec7547c` and `33f9dd83f5918ba5f5eabd86b78759df1197a9ac`.
- Failing tests due to floating point variations in calculating parameter errors. The comparison tolerances
  had to be loosened. See: `e668599966df447e99de7c145ad6c97ce7d2d814` and
  `f2729a4c5ed517b25fd99019f71ccabf65bd2ac5`.
- Indicate that the package is typed. The typing information is already there - it just didn't know it
  propagate it when running `mypy` on other packages. See: `b113e40be09705fdc3f4cce1a6ea7bbcaa14442a`.
- Updated call to the old `pachyderm.Histogram1D` API. See: `f23f31473b77203e3c2d5cf9d2617c4f3f53a6db`.

### Removed

- `evaluate_fit_component(...)` from the `ReactionPlaneFit` object. It is unnecessary given the API changes.
  See `5e2048eb1bace3d34dd428aea79b7ca002e61156`.

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

