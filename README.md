# PHY241 Project 3: Kaon Experiment

[![SPEC 7 — Seeding pseudo-random number generation](https://img.shields.io/badge/SPEC-7-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0007/)

#### Group members: [@Ion-1](https://github.com/Ion-1) [@Cookie265](https://github.com/Cookie265) [@Ar6cI](https://github.com/Ar6cI) [@AlessandroSchuetz](https://github.com/AlessandroSchuetz)

This is our repository for the group project in module PHY241 at the University of Zürich.

# Running

## Project management

This project has been configured using [uv](https://github.com/astral-sh/uv).

To run the scripts under `scripts`, make sure you have 
[uv](https://github.com/astral-sh/uv?tab=readme-ov-file#installation) 
installed and are able to run it, then run
```
cd /path/to/repository
uv run ./scripts/<script_name>.py
```
Inserting the name of the script you want to run in place of `<script_name>`.

> [!NOTE]
> Whilst this guarantees reproducibility, you can also forego it, 
> and follow what is outlined under `pyproject.toml`

## Instructions
To generate
 - a new set of values in the `value_cache.json`, simply run `generate.py` with a different `--seed`.
 - a corresponding set of plots, run `generate_plots.py`.

The code presumes the current working directory to be the repository root, due to the default
values for the location of `value_cache.json` and other files. This can be avoided by specifying
the location of the files, passed in as arguments to the script. You can run the script with
`-h` or `--help` to see the commandline options, and thus the files that need to be specified.

## Cat Placeholder
`cat.png` is generously acting as a placeholder under `./graphs`. Source:
https://github.com/Laosing/cute-cat-avatars
