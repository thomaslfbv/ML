# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Stack technique

- **Python 3.11+**
- **Reinforcement Learning** : Gymnasium, Stable-Baselines3
- **Deep Learning** : PyTorch (CUDA — GPU NVIDIA cible)
- **Environnement virtuel** : `venv/` (local au projet)

## GPU

Le projet cible une machine fixe avec GPU NVIDIA. Toujours vérifier `torch.cuda.is_available()` et utiliser `device = "cuda" if torch.cuda.is_available() else "cpu"`. Préférer les frameworks compatibles CUDA (SB3, PyTorch).

## Commandes

```bash
# Créer et activer le venv
python -m venv venv && source venv/bin/activate

# Installer les dépendances
pip install torch gymnasium stable-baselines3

# Lancer un script
python <script>.py
```
