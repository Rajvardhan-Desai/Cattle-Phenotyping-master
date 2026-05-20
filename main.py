"""Thin shim: ``python main.py`` dispatches to cattle_phenotyping.cli.main.

The real CLI lives in :mod:`cattle_phenotyping.cli`. This file exists only
because the historical entry point was ``python main.py``; new code should
use ``python -m cattle_phenotyping.cli`` or the installed
``cattle-phenotype`` console script.
"""

import sys

from cattle_phenotyping.cli import main


if __name__ == "__main__":
    sys.exit(main())
