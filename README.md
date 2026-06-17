# 🧬 Sistema OMEGA v3 — MLB Betting System

Este es el sistema predictivo OMEGA v3 para Grandes Ligas (MLB) con ensamble de modelos de Machine Learning (HGB, Random Forest, XGBoost) y módulo BETA de ajuste de ELO histórico dinámico.

## 📋 Requisitos Previos

Asegúrate de tener instalado Python 3.10 o superior.

## 🚀 Instalación y Ejecución

1. **Instalar dependencias:**
   Abre una terminal en la carpeta descomprimida y ejecuta:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ejecutar Escáner Diario:**
   Para correr las predicciones del día actual con lineups oficiales y calibración BETA, ejecuta:
   ```bash
   python3 omega_v3.py --mode predict --beta
   ```

3. **Ejecutar Backtest Histórico:**
   Para validar la efectividad histórica de los modelos sobre las temporadas 2024-2026:
   ```bash
   python3 omega_v3.py --mode backtest --beta
   ```

---
*Nota: El sistema se conecta automáticamente a la API oficial de MLB para descargar programaciones, lineups en tiempo real e históricos.*
