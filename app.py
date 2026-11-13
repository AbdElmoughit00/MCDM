import streamlit as st

import ui  # noqa: F401  enregistre le thème Plotly

st.set_page_config(page_title="Aide à la décision multicritère", page_icon=":material/balance:", layout="wide")

P = lambda path, title, icon, **kw: st.Page(f"views/{path}.py", title=title, icon=f":material/{icon}:", **kw)

nav = st.navigation({
    "": [P("home", "Accueil", "home", default=True)],
    "Pondération des critères": [
        P("ahp", "AHP", "account_tree"),
        P("bwm", "BWM", "compare_arrows"),
        P("entropy", "Entropie", "scatter_plot"),
        P("critic", "CRITIC", "grid_on"),
    ],
    "Classement additif": [
        P("waspas", "WSM, WPM, WASPAS", "functions"),
        P("topsis", "TOPSIS", "my_location"),
        P("vikor", "VIKOR", "handshake"),
    ],
    "Analyse": [P("robustness", "Robustesse et comparaison", "query_stats")],
    "Alternative unique": [P("single", "Évaluation absolue", "verified")],
    "Logique floue": [
        P("fuzzy_logic", "Ensembles flous", "blur_on"),
        P("fuzzy_mcdm", "MCDM floue et Fuzzy TOPSIS", "blur_linear"),
        P("if_topsis", "TOPSIS intuitionniste", "how_to_vote"),
    ],
})
nav.run()
