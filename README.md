# Aide à la décision multicritère

Application Streamlit qui applique pas à pas chaque méthode du cours « Aide à la décision », sur les exercices du cours et sur des cas réalistes.

| Bloc | Méthodes |
| --- | --- |
| Pondération | AHP, BWM, Entropie, CRITIC |
| Classement additif | WSM, WPM, WASPAS, TOPSIS, VIKOR |
| Surclassement | PROMETHEE I/II, ELECTRE III |
| Analyse | Spearman, sensibilité, Monte-Carlo, pipeline hybride |
| Alternative unique et IA | MAUT, seuil par ROC/Youden, ML + SHAP, MCDM dynamique |
| Flou | Ensembles flous, IFS/PFS/q-ROF, pictural, hésitant, Fuzzy TOPSIS, TOPSIS intuitionniste |
| IA générative | Scores extraits de textes (démonstration hors ligne) |

## Lancer en local

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows ; .venv/bin/pip ailleurs
.venv/Scripts/streamlit run app.py
```

## Tests

```bash
.venv/Scripts/pip install pytest
.venv/Scripts/python -m pytest
```

`tests/test_course.py` vérifie le moteur contre les résultats chiffrés du cours ; `tests/test_pages.py` exécute chaque page sur chaque jeu de données.

## Structure

- `mcdm/` : moteur de calcul en numpy, sans Streamlit, chaque étape intermédiaire est renvoyée
- `datasets.py` : exercices transcrits du cours et cas construits (graines fixes)
- `views/` : une page par méthode ; `ui.py` : éléments partagés ; `app.py` : navigation

## Écarts connus avec le cours

- **BWM, p.42** : le cours affiche les poids 0,552 / 0,207 / 0,155 / 0,086 avec ξ* = 0,08. Le programme linéaire des p.40-43 donne ξ* = 0,084, ce qui concorde, mais des poids 0,563 / 0,216 / 0,162 / 0,060. Les poids imprimés violent les contraintes du modèle (écart maximal 0,223).
- **PROMETHEE et ELECTRE III** : les diapositives n'ont que leur titre. Ce sont les formulations de référence (Brans et Vincke 1985 ; Roy 1978, s(λ) = 0,3 − 0,15 λ) qui sont implémentées.
- **MCDM dynamique** : prévision par lissage de Holt au lieu d'ARIMA/LSTM.

## Déploiement

Streamlit Community Cloud : dépôt GitHub public, fichier principal `app.py`, Python 3.13.
