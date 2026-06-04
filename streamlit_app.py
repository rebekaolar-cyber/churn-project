"""Entrypoint for Streamlit Community Cloud.

Streamlit Cloud runs the repo's root ``streamlit_app.py`` by default. This
shim puts ``src`` on the path and delegates to the package dashboard.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from churn.bootstrap import ensure_artifacts  # noqa: E402
from churn.dashboard import main  # noqa: E402

ensure_artifacts()
main()
