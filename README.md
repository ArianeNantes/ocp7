# Projet de Crédit Scoring

[![API Azure](https://github.com/ArianeNantes/ocp7/actions/workflows/main_credit-score.yml/badge.svg)](https://github.com/ArianeNantes/ocp7/actions/workflows/main_credit-score.yml)

Ce projet comprend :
- Une API FastAPI déployée sur Azure Web App
- Un dashboard Streamlit connecté à l’API, déployé sur Streamlit Cloud
- Un pipeline CI/CD avec GitHub Actions pour exécuter des tests unitaires et conditionner les déploiements

## Lancement en local

### Lancer l’API :
```bash
uvicorn main:app --reload
```
### Lancer le dashboard
```bash
PYTHONPATH=. streamlit run src/dashboard/dashboard.py
```