# Decision 0083: final data-first rendering policy

## Status

Accepted and implemented.

## Context

A sequence of rendering migrations moved biplots, prediction diagnostics, standard PLS-family
inspection, and Pi-PLS factors from package-owned plotters to direct Matplotlib construction. After
the final plotting module was removed, the repository still needed one durable policy covering
documentation, optional dependencies, examples, distribution checks, and structural enforcement.

## Decision

Pi-PLS owns numerical modeling and immutable inspection results. It does not own a plotting
submodule or public `plot_*` convenience functions. Maintained tutorials and numbered examples
render named result arrays with ordinary Matplotlib so scientific coordinates, selections, labels,
legends, layout, and output remain visible and caller-controlled.

`biplot_coordinates()` remains package-owned because balancing score and loading coordinates is a
numerical operation. Annotated biplots use Matplotlib arrows and text artists. The optional external
`textalloc` package may reposition Pulp predictor labels after final axis configuration while
treating predictor-arrow shafts and other predictor labels as obstacles; sample-score points are
not allocator obstacles. Maintained Pulp rendering falls back to ordinary Matplotlib endpoint
labels when `textalloc` is unavailable. Automatic label placement and its tuning are heuristic and
are not part of the Pi-PLS numerical contract.

Matplotlib and `textalloc` remain optional dependencies under the `examples`, `docs`, and `dev`
extras. The runtime package imports neither package. Plotting helpers may be local to the example
or renderer that owns a figure, but no shared example framework or installed plotting layer may
hide the numerical inputs or report composition.

## Consequences

- `pipls.inspection` is the public numerical analysis surface.
- Users may render results with Matplotlib, another graphics library, or no graphics system.
- Tutorials expose the result fields used for each maintained chart.
- Behavioral tests protect runtime imports without graphics dependencies, absence of a package-
  owned plotting module, and focused renderer semantics with bounded inputs. Complete tutorial
  rendering and generated-artifact integration are owned by the strict documentation target.
- Exact visual styling, source arrangement, automatically adjusted label positions, and
  incidental artist counts are not compatibility contracts.
