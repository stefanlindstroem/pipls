# Installation

PiPLS is currently installed from a source checkout. Use an isolated Python environment so its
dependencies do not interfere with other projects.

## Install the package

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

The base installation contains the regression estimators, model-selection tools, inspection
objects, and packaged reference datasets.

## Add example and plotting support

The numbered examples and tutorials use optional plotting dependencies. Install them when needed:

```bash
python -m pip install ".[examples]"
```

## Verify the installation

```bash
python -c "import pipls; print(pipls.__version__)"
```

PiPLS supports Python 3.10 through 3.14 with NumPy `>=1.26,<3`, scikit-learn `>=1.4,<2`, and
joblib `>=1.2,<2`. See the [compatibility policy](compatibility.md) for the maintained dependency
ranges and supported platforms.

Continue with the [quick start](tutorials/quick_start.md) to load an included dataset, select a
model, and obtain predictions. Contributors should instead follow the editable development setup in
the repository's
[`CONTRIBUTING.md`](https://github.com/stefanlindstroem/pipls/blob/main/CONTRIBUTING.md).
