# aptrade

[![Release](https://img.shields.io/github/v/release/vcaldas/aptrade)](https://img.shields.io/github/v/release/vcaldas/aptrade)
[![Build status](https://img.shields.io/github/actions/workflow/status/vcaldas/aptrade/main.yml?branch=main)](https://github.com/vcaldas/aptrade/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/vcaldas/aptrade/branch/main/graph/badge.svg)](https://codecov.io/gh/vcaldas/aptrade)
[![Commit activity](https://img.shields.io/github/commit-activity/m/vcaldas/aptrade)](https://img.shields.io/github/commit-activity/m/vcaldas/aptrade)
[![License](https://img.shields.io/github/license/vcaldas/aptrade)](https://img.shields.io/github/license/vcaldas/aptrade)

This is a template repository for Python projects that use uv for their dependency management.

- **Github repository**: <https://github.com/vcaldas/aptrade/>
- **Documentation** <https://vcaldas.github.io/aptrade/>


## Organization
This project is organized in a monorepo structure, with the following folders:
- `src/aptrade`: The main package of the project. This can be imported by other python projects
- `tests`: The tests for the main package.
- `docs`: The documentation for the project.
- `frontend`: The UI for the project. This is a React app that can be run with `npm start` and built with `npm run build`.
- `app`: The app for the project. This is a Flask app that provides the API routes and other backend functionalities.


## Getting started with the project

### 1. Create a New Repository

First, create a repository on GitHub with the same name as this project, and then run the following commands:

```bash
git init -b main
git add .
git commit -m "init commit"
git remote add origin git@github.com:vcaldas/aptrade.git
git push -u origin main
```

### 2. Set Up Your Development Environment

Then, install the environment and the pre-commit hooks with

```bash
make install
```

This will also generate your `uv.lock` file

### 3. Run the pre-commit hooks

Initially, the CI/CD pipeline might be failing due to formatting issues. To resolve those run:

```bash
uv run pre-commit run -a
```

### 4. Commit the changes

Lastly, commit the changes made by the two steps above to your repository.

```bash
git add .
git commit -m 'Fix formatting issues'
git push origin main
```

You are now ready to start development on your project!
The CI/CD pipeline will be triggered when you open a pull request, merge to main, or when you create a new release.

To finalize the set-up for publishing to PyPI, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/publishing/#set-up-for-pypi).
For activating the automatic documentation with MkDocs, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/mkdocs/#enabling-the-documentation-on-github).
To enable the code coverage reports, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/codecov/).

## Releasing a new version



## Inspiration and Why this project exists

This project is my attempt to combine my interests in algorithmic trading, data science, and software engineering. It is inspired by the need for a robust framework to backtest trading strategies using Python, while also providing a user-friendly interface for analysis and visualization.

This will be heavily inspired by the following projects:
Backtrader: <https://www.backtrader.com/>
Backtesting.py: <https://kernc.github.io/backtesting.py/>
pysystemtrade: <https://github.com/robcarver17/pysystemtrade>

For the real deal, please use the above projects, as they are more mature and have a larger community. This project is more of a learning experience for me, and I hope it can be useful for others as well.
