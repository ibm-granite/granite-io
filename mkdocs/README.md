# Granite-IO API Documentation with MKDocs

This directory contains the documentation site for the Granite IO project, built using [MkDocs](https://www.mkdocs.org/) and the Material for MkDocs theme.

## What is this?

- This is the source for the Granite IO documentation site.
- Documentation is written in Markdown and lives in the `docs/` subdirectory.
- The site is built and served using MkDocs, a static site generator for project documentation.

## How to Use MkDocs

### 1. Install MkDocs and the Material theme

You can install MkDocs and the Material theme using pip:

```sh
pip install mkdocs mkdocs-material
```

### 2. Serve the Documentation Locally

From the `mkdocs/` directory, run:

```sh
mkdocs serve
```

This will start a local development server (usually at http://127.0.0.1:8000/) with live reloading as you edit Markdown files.

### 3. Build the Documentation Site

To build the static site (output in the `site/` directory):

```sh
mkdocs build
```

### 4. Directory Structure

- `mkdocs.yml` — MkDocs configuration file
- `docs/` — Markdown documentation pages
- `docs/img` - Image storage **IBM Logos(s) subject to copyright**

## More Information

- See [MkDocs documentation](https://www.mkdocs.org/) for advanced usage and deployment options.