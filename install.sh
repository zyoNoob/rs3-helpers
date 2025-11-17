#!/bin/bash
uv sync --index-strategy unsafe-best-match
uv pip install -e ./x11-window-interactor
uv pip install -e ./scale-invariant-template-matching