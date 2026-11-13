# Aide à la décision multicritère

Application Streamlit qui applique pas à pas les méthodes du cours « Aide à la décision », sur les exercices du cours et sur des cas réalistes.

| Bloc | Méthodes |
| --- | --- |
| Pondération | AHP, BWM, Entropie, CRITIC |
| Classement | WSM, WPM, WASPAS, TOPSIS, VIKOR |
| Analyse | Spearman, sensibilité, Monte-Carlo |
| Alternative unique | MAUT, seuil par ROC/Youden, distance à l'idéal |
| Flou | Ensembles flous, IFS/PFS/q-ROF, pictural, hésitant, Fuzzy TOPSIS, TOPSIS intuitionniste |

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
- `datasets.py` : exercices du cours et cas construits (graines fixes)
- `views/` : une page par méthode ; `ui.py` : éléments partagés ; `app.py` : navigation

## Déploiement

Streamlit Community Cloud : dépôt GitHub public, fichier principal `app.py`, Python 3.13.
