# An Introduction to Machine Learning

This repository holds the Jupyter Book source for An Introduction to Machine Learning.

## To make a change to the book and update `https://joaomh.github.io/ml-book/`

1. Get your copy of this repository:

   ```
   git clone https://github.com/joaomh/ml-book
   ```
2. Change the file you wish and commit it to the repository.
3. Push your change back to the `ml-book` repository (ideally via a pull request).
4. That's it a GitHub Action will build the book and deploy it to [ml-book](https://joaomh.github.io/ml-book/intro.html)

## How this repository is deployed

* When you make a change to this repository and push it to the `main` branch, the book's HTML will automatically be pushed.
* This process is handled by [this GitHub Action](.github/workflows/deploy.yml)

## Build and preview the text locally

To build locally, `pip install -r requirements.txt` and then `jupyter-book build ml-book`

**Follow the build instructions on the Jupyter Book guide**. The guide has information on how to use the Jupyter Book CLI to build this book. You can find the [Jupyter Book build instructions here](https://jupyterbook.org/start/build.html).

```
ghp-import -n -p -f _build/html
```
