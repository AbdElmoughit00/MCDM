"""Exécute chaque page sans navigateur et vérifie qu'aucune exception ni erreur n'apparaît."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

VIEWS = sorted((Path(__file__).resolve().parents[1] / "views").glob("*.py"))


@pytest.mark.parametrize("view", VIEWS, ids=[v.stem for v in VIEWS])
def test_page_runs(view):
    at = AppTest.from_file(str(view), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    # Chaque jeu de données et chaque variante (radio) doit aussi s'exécuter sans erreur
    for kind in ("selectbox", "radio"):
        for i in range(len(getattr(at, kind))):
            widget = getattr(at, kind)[i]
            if kind == "selectbox" and widget.label != "Jeu de données":
                continue
            for option in widget.options:
                if option == "Importer un CSV":
                    continue
                getattr(at, kind)[i].set_value(option).run()
                assert not at.exception, (widget.label, option, [e.value for e in at.exception])
                assert not at.error, (widget.label, option, [e.value for e in at.error])
