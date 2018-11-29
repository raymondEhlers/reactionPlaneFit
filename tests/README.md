# Testing the Reaction Plane Fit package

Tests are implemented using `pytest`. To execute the testing, I tend to use something like:

```bash
$ pytest -l --cov=reaction_plane_fit --cov-report html --cov-branch --durations=5 --mpl tests/
```

This assumes you are running from the root repository folder and will report on which tests are the slowest
as well as provide a coverage report for (in this case) the entire `reaction_plane_fit` module. The branch
coverage is particularly useful to help avoid missing coverage over control flow.

There are two particularly noteworthy classes of marked tests:

- `ROOT`: Tests which require ROOT.
- `slow`: Tests which are slow, as they run the entire fit code. These are particularly slow because of the
  error calculation.

Recall that a particular class of tests (named `CLASS` in this example) can be skipped by passing `-m "not
CLASS"` argument to `pytest`.

## Installing the test dependencies

Beyond ROOT (which is required for tests marked as "ROOT"), the package test dependencies can be installed
with:

```bash
$ pip install -e .[tests]
```

## Image comparisons

Image comparisons are performed via `pytest-mpl`. To generate the comparison images, run in the `tests`
directory:

```bash
$ pytest --mpl-generate-path=baseline
```

This will create comparison images in the `tests/baseline` directory. Once the comparisons have been created,
you can compare the images by passing the option `--mpl` to `pytest`. Note that if the argument is not passed,
the test code itself will be run, but the image won't be compared.

Note that if a new figure is added, it is up to you to generate and add the baseline image locally. Travis
will only do the comparison, not the generation.
