# Examples

The six numbered scripts form a progression from a short automatic fit to explicit model selection,
method comparison, and complete real-data analyses. Each example is meant to add one main idea. Run
the scripts when you want the numerical details, figures, and exact configuration; this page is only
a guide to what each example teaches.

For the underlying API, see the [path-selection reference](api/path.md). For the packaged real
datasets and their provenance, see the [dataset guide](datasets.md).

## Choose an example

1. **`01_pulp_quick_start.py` — start with the shortest complete workflow.** Search, select by a CV
   rule, refit, and inspect fitted predictions on [Pulp](datasets.md#pulp-real-data-integration).
   The [Quick Start tutorial](tutorials/quick_start.md) follows this example.

2. **`02_synthetic_path_selection.py` — inspect before selecting.** Examine the component path, make
   an explicit component-count choice, inspect the corresponding predictor-rank evidence, refit that
   selection, and assess it on independent test data. See the
   [synthetic tutorial](tutorials/synthetic.md).

3. **`03_pls_path_comparison.py` — compare PLS-family component paths.** Compare the two Π-PLS
   response-subspace policies with ordinary PLS on the same validation splits. This is a comparison
   workflow rather than a model-selection recipe; see
   [Compare response-subspace policies](#compare-response-subspace-policies).

4. **`04_pulp_real_data.py` — carry out a complete manual Π-PLS analysis.** Move from component-path
   review and explicit selection to out-of-fold review, refitting, and model interpretation. The
   [Pulp tutorial](tutorials/pulp.md) is the guided version.

5. **`05_sugarcane_real_data.py` — regularize a spectral predictor subspace explicitly.** Fix
   $r_\pi$ with the EPV policy, then choose the component
   count separately and interpret the retained wavelength directions on
   [Sugarcane](datasets.md#sugarcane-spectral-integration). See the
   [EPV policy](path_selection.md#epv-policy) and the distinct roles of
   [$r_\pi$ and $h$](theory.md#interpretation-of-the-ranks).

6. **`06_tobacco_real_data.py` — regularize spectral rank from validation evidence.** Optimize
   $r_\pi$, but use a 20% relative tolerance to prefer a smaller retained predictor subspace when
   its CV performance remains close to the conditional optimum. Apply component-count parsimony in
   a separate tolerance stage, then carry the selection through out-of-fold review and refitting on
   [Tobacco](datasets.md#tobacco-spectral-integration). See
   [search-owned selection rules](path_selection.md#search-owned-selection-rules).

## Compare response-subspace policies {#compare-response-subspace-policies}

Example 03 compares the default `"cross_covariance"` Π-PLS policy, the `"least_squares"` software
extension, and ordinary PLS while holding the validation splits fixed. It uses the three packaged
real datasets and one deliberately difficult synthetic case. It does not fit a final model or claim
that one method is generally superior.

The two Π-PLS constructions are described under
[Response-subspace selection](theory.md#response-subspace-selection). Cross-covariance is the
construction in the [companion paper](citation.md#companion-paper); least squares is a software
extension.

## Complete real-data analyses {#complete-real-data-analyses}

Examples 04--06 are the longer reference analyses. They share the same broad sequence—review the
search path, create one selection, inspect evidence for that selection, refit it on all observations,
and inspect the fitted model—but differ in the kind of data and the selection strategy. Dataset
sources, licenses, and preparation are documented in the [dataset guide](datasets.md).

## Run the examples

From the repository root:

```bash
python -m pip install ".[examples]"
python examples/01_pulp_quick_start.py
```

Generated files are written below `examples/results/`. Run the complete maintained set with:

```bash
make examples
```

The longer examples are separate from `make check` because they are application workflows rather
than package tests.

The component paths and same-search out-of-fold reports used here are model-development evidence,
not independent post-selection validation. Applications with grouped, temporal, or otherwise
structured data should supply an appropriate splitter. See
[Path and selection](path_selection.md) for search and selection contracts and
[OOF diagnostics](oof_diagnostics.md) for selection-conditioned OOF reporting.
