---
html_theme.sidebar_secondary.remove: true
---

# Aptrade

```{include} ../README.md
:start-after: "# Aptrade"
```


How the documentation is structured
-----------------------------------


<!-- https://sphinx-design.readthedocs.io/en/latest/grids.html -->

::::{grid} 2
:gutter: 4

:::{grid-item-card} {material-regular}`directions_walk;2em`
```{toctree}
:maxdepth: 2
tutorials
```
+++
Tutorials for installation and typical usage. New users start here.
:::

:::{grid-item-card} {material-regular}`directions;2em`
```{toctree}
:maxdepth: 2
how-to
```
+++
Practical step-by-step guides for the more experienced user.
:::

:::{grid-item-card} {material-regular}`info;2em`
```{toctree}
:maxdepth: 2
components
```
+++
Explanations of how it works and why it works that way.
:::

:::{grid-item-card} {material-regular}`menu_book;2em`
```{toctree}
:maxdepth: 2
reference
```
+++
Technical reference material including APIs and release notes.
:::

::::
