# 24-Hour Ahead Power Forecasting with Temporal Fusion Transformer (TFT)

## 🎯 Obiettivo del Progetto

Previsione della produzione fotovoltaica nelle prossime 24 ore utilizzando l'architettura **Temporal Fusion Transformer (TFT)** su 2 anni di dati storici di produzione e variabili meteorologiche.

## 📊 Dataset

Il progetto utilizza 4 file CSV:

### Dati di Produzione PV:
- `pv_dataset (1).xlsx - 07-10--06-11.csv`
- `pv_dataset (1).xlsx - 07-11--06-12.csv`

### Dati Meteorologici:
- `wx_dataset.xlsx - 07-10--06-11.csv`
- `wx_dataset.xlsx - 07-11--06-12.csv`

**Note sui dati:**
- Colonna timestamp PV: "Max kWp" → rinominata in "datetime"
- Colonna target: nome numerico (es. "82.41") → rinominata in "power_kw"
- Colonna timestamp meteo: "dt_iso" → rinominata in "datetime"

## 🛠️ Tecnologie Utilizzate

- **Python 3.8+**
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **PyTorch** - Deep learning framework
- **PyTorch Lightning** - Training orchestration
- **pytorch-forecasting** - Time series forecasting library
- **matplotlib/seaborn** - Data visualization

## 📦 Installazione

```bash
# Crea un ambiente virtuale (consigliato)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oppure
venv\Scripts\activate  # Windows

# Installa le dipendenze
pip install -r requirements.txt
```

## 🚀 Utilizzo

Apri il notebook Jupyter:

```bash
jupyter lab PV_Forecasting_TFT.ipynb
```

Oppure usa VS Code con l'estensione Jupyter.

## 📝 Pipeline del Progetto

### 1. **Data Loading & Merging**
   - Caricamento dei 4 file CSV
   - Concatenazione dei dati per anno
   - Conversione timestamp e gestione timezone
   - Merge tra dati PV e meteo
   - Rimozione duplicati temporali

### 2. **Data Analysis & Missing Values**
   - Analisi esplorativa dei dati
   - Gestione valori mancanti:
     - `rain_1h`: NaN → 0
     - Altre colonne: interpolazione lineare limitata

### 3. **Feature Engineering**
   - Estrazione feature temporali: `hour`, `day_of_month`, `month`, `day_of_week`
   - Creazione `group_id` = "PV1"
   - Generazione `time_idx` (indice temporale incrementale)
   - Conversione a float32

### 4. **Pre-processing per TFT**
   - Configurazione `TimeSeriesDataSet`:
     - **Encoder length**: 168 ore (1 settimana di storico)
     - **Prediction length**: 24 ore (giorno successivo)
     - **Known reals**: variabili temporali + meteo
     - **Unknown reals**: `power_kw`
   - Split Train/Validation (ultimi 3 mesi per validation)
   - DataLoaders configurati

### 5. **Modeling - Temporal Fusion Transformer**
   - Architettura TFT con:
     - `hidden_size`: 128
     - `lstm_layers`: 2
     - `attention_head_size`: 4
     - `dropout`: 0.1
   - Loss function: **QuantileLoss** (intervalli di confidenza)

### 6. **Training**
   - PyTorch Lightning Trainer
   - Early Stopping (patience=10)
   - Learning Rate Monitoring
   - TensorBoard logging
   - Max epochs: 50

### 7. **Evaluation & Visualization**
   - Metriche: MAE, RMSE, MAPE, R²
   - Metriche per horizon (1-24 ore)
   - Visualizzazioni:
     - Predizioni vs valori reali
     - Scatter plot
     - Distribuzione errori
     - Serie temporale continua
   - Interpretabilità (attention weights, variable importance)

## 📈 Risultati

I risultati vengono salvati automaticamente in:
- `tft_results.json` - Metriche di performance
- `tft_predictions.csv` - Predizioni complete
- `lightning_logs/` - TensorBoard logs

### Visualizzazione con TensorBoard

```bash
tensorboard --logdir=lightning_logs
```

## 📊 Struttura del Progetto

```
DL-Project-TFT/
├── PV_Forecasting_TFT.ipynb    # Notebook principale
├── requirements.txt             # Dipendenze
├── README.md                    # Questo file
├── pv_dataset (1).xlsx - 07-10--06-11.csv
├── pv_dataset (1).xlsx - 07-11--06-12.csv
├── wx_dataset.xlsx - 07-10--06-11.csv
├── wx_dataset.xlsx - 07-11--06-12.csv
├── tft_results.json            # Risultati (generato)
├── tft_predictions.csv         # Predizioni (generato)
└── lightning_logs/             # Training logs (generato)
```

## 🔧 Configurazione Modello

### Parametri Principali:
- **Encoder**: 168 ore (1 settimana)
- **Decoder**: 24 ore (1 giorno)
- **Hidden size**: 128
- **Attention heads**: 4
- **LSTM layers**: 2
- **Dropout**: 0.1
- **Batch size**: 64
- **Learning rate**: 0.03

### Variabili:
- **Static categorical**: `group_id`
- **Time-varying known**: `hour`, `day_of_month`, `month`, `day_of_week`, `temp`, `Dni`, `Ghi`, `humidity`, `clouds_all`, ecc.
- **Time-varying unknown**: `power_kw` (target)

## 📚 Riferimenti

- **Temporal Fusion Transformer**: [Lim et al., 2021](https://arxiv.org/abs/1912.09363)
- **PyTorch Forecasting**: [Documentation](https://pytorch-forecasting.readthedocs.io/)
- **PyTorch Lightning**: [Documentation](https://lightning.ai/docs/pytorch/stable/)

## 🤝 Autore

Progetto sviluppato per il corso di Deep Learning - Magistrale UCBM

## 📄 Licenza

Questo progetto è fornito a scopo educativo e di ricerca.

## ⚠️ Note Importanti

1. **Memoria GPU**: Il modello richiede una GPU con almeno 8GB di VRAM per il training ottimale
2. **Tempo di training**: ~30-50 minuti su GPU, diverse ore su CPU
3. **File CSV**: Assicurati che i 4 file CSV siano nella stessa directory del notebook
4. **Timezone**: I timestamp vengono convertiti in timezone-naive per allineamento

## 🐛 Troubleshooting

### Errore: "FileNotFoundError"
- Verifica che i 4 file CSV siano nella directory corretta
- Controlla i nomi esatti dei file

### Errore: "CUDA out of memory"
- Riduci il `batch_size` (prova 32 o 16)
- Riduci `hidden_size` (prova 64)

### Warning: "No CUDA device detected"
- Il training avverrà su CPU (più lento ma funzionale)
- Considera l'uso di Google Colab per accesso GPU gratuito

## 📞 Supporto

Per domande o problemi, apri una issue nel repository.