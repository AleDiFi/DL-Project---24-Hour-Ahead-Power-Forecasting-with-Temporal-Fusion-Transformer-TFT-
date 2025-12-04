# Aggiornamenti Notebook PV_Forecasting_TFT.ipynb

## 🎯 Modifiche Implementate

### 1. **Temporal Cross-Validation** (sostituzione dell'holdout)
- ✅ Implementata **5-fold temporal cross-validation** invece del semplice train/validation split
- ✅ Split temporale bilanciato: 80% training / 20% validation per ogni fold
- ✅ Prevenzione data leakage: i fold rispettano l'ordine temporale
- ✅ Crescita progressiva del training set tra i fold

**Cella aggiunta:** Sezione 5 - Setup Temporal Cross-Validation
```python
folds = setup_temporal_cross_validation(data, n_folds=5, val_ratio=0.2)
```

### 2. **Hyperparameter Tuning con Optuna**
- ✅ **9 iperparametri ottimizzati**:
  - `hidden_size`: [64, 128, 192, 256]
  - `lstm_layers`: [1, 2, 3]
  - `attention_head_size`: [1, 2, 4, 8]
  - `hidden_continuous_size`: [8, 16, 32]
  - `dropout`: [0.1, 0.4]
  - `learning_rate`: [1e-4, 1e-1] (log scale)
  - `batch_size`: [32, 64, 128]
  - `patience`: [10, 30]
  - `gradient_clip_val`: [0.1, 2.0]

- ✅ **Algoritmo**: TPE Sampler (Tree-structured Parzen Estimator)
- ✅ **Pruning**: Median Pruner per terminare trial non promettenti
- ✅ **Metric**: Media della validation loss sui 5 fold (robustezza)

**Celle aggiunte:**
- Sezione 5.1: Hyperparameter Search Space
- Sezione 5.2: Objective Function con Cross-Validation
- Sezione 5.3: Avvio Optuna Optimization
- Sezione 5.4: Analisi Risultati Optimization

### 3. **Training Finale con Best Hyperparameters**
- ✅ Training completo (150 epochs) dopo l'optimization
- ✅ Usa i migliori iperparametri trovati da Optuna
- ✅ Training sull'ultimo fold (massimo dataset disponibile)
- ✅ Early stopping con patience ottimizzata

**Celle modificate:**
- Sezione 6: Training Modello Finale (riscritta)
- Sezione 7: Training del Modello Finale (aggiornata)

### 4. **Salvataggio Completo del Modello**
- ✅ **Directory `final_best_model/`** con 3 file:
  1. `best_tft_model.ckpt` - Checkpoint PyTorch Lightning
  2. `best_hyperparameters.json` - Iperparametri ottimali
  3. `model_info.json` - Metadata completi (timestamp, features, training info)

- ✅ **File di output aggiuntivi**:
  - `final_results.json` - Metriche finali (MAE, RMSE, R², MAPE)
  - `optuna_study_results.json` - Risultati optimization completi
  - `training_report_[timestamp].txt` - Report testuale dettagliato
  - `final_predictions.csv` - Predizioni per analisi

**Cella aggiunta:** Sezione 8 - Salvataggio modello e metadata

### 5. **Visualizzazioni Optimization**
- ✅ **4 plot per analisi optimization**:
  1. Optimization History (convergenza trial)
  2. Parameter Importance (feature importance)
  3. Learning Rate vs Validation Loss
  4. Hidden Size vs Validation Loss

- ✅ Top 5 trials visualizzati
- ✅ Statistiche complete (completed/failed/pruned trials)

**Cella aggiunta:** Sezione 5.4 - Plot analisi risultati

### 6. **Rimozione Celle Obsolete**
- ❌ Rimossa cella identificazione features (ora in setup CV)
- ❌ Rimossa cella split train/validation hardcoded
- ❌ Rimossa cella analisi distribuzione target diagnostica
- ❌ Rimossa cella creazione TimeSeriesDataSet manuale
- ❌ Rimossa cella creazione DataLoaders manuale

**Motivo**: Queste operazioni sono ora gestite automaticamente nella funzione `objective()` durante l'optimization e nel training finale.

---

## 📊 Workflow Completo

### **PRIMA** (Holdout Semplice):
```
Data Loading → Feature Engineering → Train/Val Split (fisso) 
→ TimeSeriesDataSet → Model Config (manuale) → Training 
→ Evaluation → Save Results
```

### **ADESSO** (CV + Hyperparameter Tuning):
```
Data Loading → Feature Engineering → Temporal Cross-Validation Setup 
→ Optuna Hyperparameter Search (5-fold CV per trial) 
→ Best Hyperparameters Selection 
→ Final Model Training (150 epochs) 
→ Model Saving (checkpoint + metadata) 
→ Evaluation & Visualization 
→ Complete Report Generation
```

---

## 🔧 Parametri Configurabili

### Nel notebook principale:
```python
# Sezione 5.3 - Avvio Optimization
N_TRIALS = 20  # 🔧 MODIFICARE QUI: Numero di trial Optuna

# Nella funzione setup_temporal_cross_validation
n_folds = 5      # Numero di fold per CV
val_ratio = 0.2  # Percentuale validation set (20%)

# Nella funzione objective
max_epochs = 30  # Epochs per fold durante optimization

# Nel training finale
max_epochs = 150 # Epochs per modello finale
```

---

## 🚀 Come Usare il Notebook Aggiornato

### 1. **Eseguire tutte le celle in ordine**:
   - Celle 1-4: Data loading, preprocessing, feature engineering (invariate)
   - Cella 5: Setup temporal cross-validation
   - Celle 5.1-5.3: Hyperparameter optimization con Optuna
   - Cella 5.4: Analisi risultati optimization
   - Celle 6-7: Training modello finale con best hyperparameters
   - Cella 8: Salvataggio modello + metadata
   - Celle 9-12: Evaluation, visualizations, interpretability

### 2. **Modificare numero di trials** (opzionale):
   ```python
   # Nella cella 5.3
   N_TRIALS = 50  # Aumenta per ottimizzazione più approfondita
   ```

### 3. **Controllare output files**:
   - `final_best_model/` - Modello pronto per deployment
   - `final_results.json` - Metriche per pubblicazione
   - `training_report_[timestamp].txt` - Documentazione completa

---

## 📈 Vantaggi delle Modifiche

### 1. **Robustezza**
- ✅ Cross-validation riduce overfitting
- ✅ Validazione su 5 fold diversi garantisce generalizzazione
- ✅ Prevenzione data leakage con split temporale

### 2. **Ottimizzazione**
- ✅ Hyperparameter tuning automatico (no trial-and-error manuale)
- ✅ Bayesian optimization più efficiente del grid search
- ✅ Pruning intelligente risparmia tempo di training

### 3. **Riproducibilità**
- ✅ Tutti gli hyperparameters salvati automaticamente
- ✅ Seed fisso (42) per Optuna sampler
- ✅ Metadata completi per ogni esperimento

### 4. **Production-Ready**
- ✅ Modello salvato in formato standard (PyTorch Lightning)
- ✅ Configurazione completa esportata in JSON
- ✅ Report testuale per documentazione

---

## ⚠️ Note Importanti

1. **Tempo di esecuzione**: 
   - 20 trials × 5 fold × ~30 epochs ≈ **2-4 ore** (con RTX 4060)
   - Ridurre `N_TRIALS` per test veloci (es. 5 trials)

2. **Memoria GPU**:
   - Monitoring attivo durante optimization
   - Cleanup automatico dopo ogni fold
   - Batch size ottimizzato automaticamente da Optuna

3. **Compatibilità**:
   - Richiede `optuna` installato: `pip install optuna`
   - Compatibile con PyTorch Lightning 2.x
   - Testato su Windows con Python 3.13

4. **File generati**:
   - Tutti i file vengono salvati nella directory del notebook
   - `lightning_logs/` contiene i log TensorBoard per ogni trial
   - Pulire periodicamente `lightning_logs/` per liberare spazio

---

## 📝 Prossimi Passi Suggeriti

### Opzionali (non implementati):
1. **Test Set separato**: Aggiungere un test set finale (15% del dataset) per valutazione imparziale
2. **Ensemble models**: Combinare i top-3 modelli per predizioni più robuste
3. **Deployment**: Creare API REST per servire il modello in produzione
4. **Monitoring**: Dashboard per monitorare performance in tempo reale
5. **Retraining automatico**: Pipeline per retraining periodico con nuovi dati

---

## 🎓 Riferimenti

- **Optuna Documentation**: https://optuna.readthedocs.io/
- **PyTorch Forecasting**: https://pytorch-forecasting.readthedocs.io/
- **Temporal Fusion Transformer Paper**: https://arxiv.org/abs/1912.09363
- **Time Series Cross-Validation**: https://robjhyndman.com/hyndsight/tscv/

---

**Ultima modifica**: 2025-12-01
**Versione**: 2.0 (con Hyperparameter Optimization)
**Autore**: GitHub Copilot
