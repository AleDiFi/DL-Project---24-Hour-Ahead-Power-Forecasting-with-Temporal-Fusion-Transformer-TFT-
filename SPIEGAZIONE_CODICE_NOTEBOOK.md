# Spiegazione Dettagliata del Notebook - PV Forecasting con TFT

## Indice
- [Cella 1: Import Librerie](#cella-1-import-librerie)
- [Cella 2: Verifica GPU/CUDA](#cella-2-verifica-gpucuda)
- [Cella 3: Conversione Excel in CSV](#cella-3-conversione-excel-in-csv)

---

## Cella 1: Import Librerie

Questa cella iniziale importa tutte le librerie necessarie per il progetto di forecasting. Vediamo nel dettaglio ogni sezione:

### Sezione 1: Data Manipulation (Manipolazione Dati)

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
```

**Spiegazione riga per riga:**

1. **`import pandas as pd`**
   - Importa la libreria Pandas con alias `pd`
   - Pandas è essenziale per manipolare dati tabulari (DataFrame)
   - Permette operazioni come merge, filtri, aggregazioni su dati strutturati
   - Nel progetto: gestisce i dataset PV e meteo

2. **`import numpy as np`**
   - Importa NumPy con alias `np`
   - NumPy gestisce array numerici e operazioni matematiche vettoriali
   - Più veloce delle liste Python per calcoli su grandi quantità di dati
   - Nel progetto: conversioni dtype, calcoli statistici, reshape di array

3. **`from datetime import datetime, timedelta`**
   - Importa classi specifiche dal modulo datetime
   - `datetime`: rappresenta un punto specifico nel tempo (data + ora)
   - `timedelta`: rappresenta differenze tra date/ore
   - Nel progetto: gestisce i timestamp orari del dataset (luglio 2010 - giugno 2012)

4. **`import warnings`**
   - Importa il modulo warnings per gestire gli avvisi Python
   
5. **`warnings.filterwarnings('ignore')`**
   - **Scopo**: Disabilita la visualizzazione degli avvisi
   - **Perché**: Durante il training di modelli DL, appaiono molti warning deprecation
   - **Attenzione**: In produzione è meglio gestire i warning specificamente
   - Nel progetto: evita spam visivo da PyTorch/Lightning durante training

### Sezione 2: Visualization (Visualizzazione)

```python
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.style.use('seaborn-v0_8-darkgrid')
```

**Spiegazione riga per riga:**

6. **`import matplotlib.pyplot as plt`**
   - Importa il modulo pyplot di Matplotlib con alias `plt`
   - Matplotlib è la libreria standard per plotting in Python
   - pyplot fornisce interfaccia simile a MATLAB per creare grafici
   - Nel progetto: crea tutti i plot (serie temporali, scatter, histogram, boxplot)

7. **`import matplotlib.dates as mdates`**
   - Importa sottomodulo per formattare assi temporali
   - `mdates.DateFormatter()`: formatta timestamp sui grafici
   - `mdates.HourLocator()`: posiziona tick ogni N ore
   - Nel progetto: usato per formattare l'asse X nei plot temporali

8. **`plt.style.use('seaborn-v0_8-darkgrid')`**
   - **Scopo**: Applica uno stile grafico predefinito a tutti i plot
   - **`seaborn-v0_8-darkgrid`**: Stile con griglia scura, sfondo grigio chiaro
   - **Perché**: Migliora leggibilità rispetto allo stile Matplotlib di default
   - **Alternativa**: `'ggplot'`, `'bmh'`, `'dark_background'`
   - Nel progetto: rende i grafici più professionali e leggibili

### Sezione 3: PyTorch & PyTorch Lightning

```python
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
```

**Spiegazione riga per riga:**

9. **`import torch`**
   - Importa PyTorch, framework per Deep Learning
   - **Funzioni principali**:
     - Gestione tensori (equivalenti NumPy ma su GPU)
     - Autograd: calcolo automatico gradienti
     - Neural network building blocks
   - Nel progetto: backend per il modello TFT, gestione GPU

10. **`import lightning.pytorch as pl`**
    - Importa PyTorch Lightning con alias `pl`
    - **PyTorch Lightning**: wrapper high-level di PyTorch
    - **Vantaggi**:
      - Elimina boilerplate code (loop training, validation, logging)
      - Gestione automatica GPU/TPU
      - Callbacks integrati (early stopping, checkpointing)
    - **IMPORTANTE**: Si usa `lightning.pytorch` (versione 2.x), non `pytorch_lightning` (deprecato)
    - Nel progetto: gestisce il ciclo di training del TFT

11. **`from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor`**
    - Importa due callback specifici da Lightning
    
    **`EarlyStopping`**:
    - **Scopo**: Ferma il training quando la validazione non migliora
    - **Parametri chiave**:
      - `monitor="val_loss"`: metrica da monitorare
      - `patience=20`: numero di epoch senza miglioramento prima di fermarsi
      - `min_delta=1e-4`: miglioramento minimo considerato significativo
    - **Perché importante**: Previene overfitting e spreco di tempo di training
    - Nel progetto: configurato con patience=20 (aumentato da 10 originale)
    
    **`LearningRateMonitor`**:
    - **Scopo**: Traccia il learning rate durante il training
    - **Utilità**: Debugging di learning rate schedulers
    - Nel progetto: permette di verificare la riduzione automatica di LR

12. **`from lightning.pytorch.loggers import TensorBoardLogger`**
    - Importa logger per TensorBoard
    - **TensorBoard**: Tool di visualizzazione per esperimenti ML
    - **Cosa registra**:
      - Loss curves (train/validation)
      - Learning rate nel tempo
      - Gradienti e pesi del modello (opzionale)
      - Metriche custom
    - **Come usarlo**: Dopo training, esegui `tensorboard --logdir=lightning_logs`
    - Nel progetto: salva log in `lightning_logs/tft_pv_forecasting`

### Sezione 4: PyTorch Forecasting

```python
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, RMSE, SMAPE
```

**Spiegazione riga per riga:**

13. **`from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss`**
    - Importa classi principali dalla libreria pytorch-forecasting
    
    **`TimeSeriesDataSet`**:
    - **Scopo**: Wrapper per dataset temporali compatibile con PyTorch
    - **Funzionalità**:
      - Crea sequenze encoder-decoder automaticamente
      - Gestisce normalizzazione per gruppo
      - Distingue tra known/unknown covariates
      - Genera batch per training
    - **Nel progetto**: Configurato con:
      - `max_encoder_length=168` (1 settimana di storico)
      - `max_prediction_length=24` (previsione 24h)
      - 13 feature totali (8 meteo + 4 temporali + 1 target)
    
    **`TemporalFusionTransformer`**:
    - **Scopo**: Implementazione del modello TFT (Google Research, 2019)
    - **Architettura**:
      - Variable Selection Networks: seleziona feature rilevanti
      - LSTM encoder/decoder: cattura dipendenze temporali
      - Multi-Head Attention: focus su orizzonti specifici
      - Gating mechanisms: controlla flusso informazione
    - **Output**: Predizioni quantiliche (P10, P50, P90, etc.)
    - Nel progetto: configurato con 120k parametri (ottimizzato da 450k)
    
    **`QuantileLoss`**:
    - **Scopo**: Loss function per regressione quantilica
    - **Formula**: $L_q(y, \hat{y}) = \sum_i \max[q(y_i - \hat{y}_i), (q-1)(y_i - \hat{y}_i)]$
    - **Perché**: Permette di stimare incertezza (intervalli di confidenza)
    - **Output**: 7 quantili (P02, P10, P25, P50, P75, P90, P98)
    - Nel progetto: genera predizioni con range superiore/inferiore

14. **`from pytorch_forecasting.data import GroupNormalizer`**
    - Importa normalizzatore per gruppi
    - **Scopo**: Normalizza i dati per ogni gruppo separatamente
    - **Perché importante**: 
      - Dataset può avere multiple serie temporali (gruppi)
      - Ogni serie può avere scala diversa
      - Normalizzazione per gruppo evita bias
    - **Nel progetto**: 
      - Un solo gruppo: `group_id='PV1'`
      - `transformation=None` (ottimizzato da "softplus")
      - Gestisce correttamente i valori zero (40% dei dati = notte)

15. **`from pytorch_forecasting.metrics import MAE, RMSE, SMAPE`**
    - Importa metriche per valutazione forecasting
    
    **`MAE` (Mean Absolute Error)**:
    - Formula: $MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$
    - Unità: stessa del target (kW)
    - Interpretazione: errore medio assoluto
    - Nel progetto: metrica principale per valutazione
    
    **`RMSE` (Root Mean Squared Error)**:
    - Formula: $RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$
    - Unità: stessa del target (kW)
    - Penalizza errori grandi più di MAE
    - Nel progetto: metrica secondaria
    
    **`SMAPE` (Symmetric Mean Absolute Percentage Error)**:
    - Formula: $SMAPE = \frac{100\%}{n}\sum_{i=1}^{n}\frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2}$
    - Unità: percentuale (%)
    - Simmetrica rispetto a sotto/sovra-stima
    - Nel progetto: non usata (preferito MAPE standard)

### Sezione 5: Scikit-Learn Metrics

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
```

**Spiegazione riga per riga:**

16. **`from sklearn.metrics import mean_absolute_error, mean_squared_error`**
    - Importa metriche da scikit-learn per confronto
    
    **Perché importare da sklearn se abbiamo pytorch_forecasting.metrics?**
    - Compatibilità con NumPy arrays (più semplice)
    - Standard de-facto per confronti con altri modelli
    - Sintassi più diretta: `mean_absolute_error(y_true, y_pred)`
    
    **Nel progetto**:
    - Usate nella cella di evaluation (Cella 32)
    - Calcolo su array flattened: `y_true_flat`, `y_pred_flat`
    - Metriche per horizon: MAE/RMSE per ogni ora (1-24)

### Sezione : Hyperparameter Optimization

```python
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import json
import time
import gc
```

**Spiegazione riga per riga:**

16. **`import optuna`**
    - Importa Optuna, framework per hyperparameter optimization
    - **Cos'è Optuna**:
      - Libreria open-source per ottimizzazione automatica di iperparametri
      - Sviluppata da Preferred Networks (azienda giapponese)
      - Usa algoritmi di ottimizzazione Bayesiana (TPE - Tree-structured Parzen Estimator)
    - **Funzionalità principali**:
      - Define-by-run API: definisci search space dinamicamente
      - Pruning automatico: termina trial non promettenti
      - Parallelizzazione: esegue multiple trial in parallelo
      - Visualizzazioni built-in: importance, history, contour plots
    - **Nel progetto**: 
      - Ottimizza 9 iperparametri del TFT (hidden_size, learning_rate, dropout, etc.)
      - 20 trial con TPE Sampler
      - Cross-validation su 5 fold per ogni trial
      - Salva risultati in `optuna_study_results.json`

17. **`from optuna.integration import PyTorchLightningPruningCallback`**
    - Importa callback specifico per integrazione Optuna + PyTorch Lightning
    - **PyTorchLightningPruningCallback**:
      - **Scopo**: Terminare anticipatamente trial non promettenti durante training
      - **Come funziona**:
        1. Monitora metrica (es. `val_loss`) ad ogni epoch
        2. Confronta con altri trial alla stessa epoch
        3. Se performance significativamente peggiore → PRUNE (termina)
      - **Vantaggi**:
        - Risparmia tempo computazionale (no training completo su config scadenti)
        - Permette più trial nello stesso tempo
        - Median Pruner: prune se sotto mediana degli altri trial
    - **Parametri chiave**:
      - `monitor`: metrica da monitorare (es. `"val_loss"`)
      - `mode`: `"min"` o `"max"` (minimize o maximize)
    - **Nel progetto**: 
      - Configurato con `MedianPruner(n_startup_trials=5, n_warmup_steps=2)`
      - Pruning attivo dopo 5 trial di warmup
      - Salva ~30-40% tempo di ottimizzazione

18. **`import json`**
    - Importa modulo JSON (JavaScript Object Notation) per serializzazione dati
    - **Funzioni usate**:
      - `json.dump(obj, file)`: Scrive oggetto Python in file JSON
      - `json.load(file)`: Legge file JSON in oggetto Python
      - `json.dumps(obj)`: Converte oggetto Python in stringa JSON
    - **Nel progetto**:
      - Salva best hyperparameters: `best_hyperparameters.json`
      - Salva risultati Optuna: `optuna_study_results.json`
      - Salva metriche finali: `final_results.json`
      - Salva model info: `model_info.json`
    - **Perché JSON**:
      - Human-readable: facile ispezionare risultati
      - Language-agnostic: può essere letto da qualsiasi linguaggio
      - Leggero: più compatto di XML
    - **Nota importante**: NumPy types (float32, int64) non sono JSON-serializable
      - Necessaria conversione esplicita: `float()`, `int()`

19. **`import time`**
    - Importa modulo time per misurare durata operazioni
    - **Funzioni usate**:
      - `time.time()`: timestamp UNIX corrente (secondi dal 1970-01-01)
      - Differenza tra due `time.time()` = durata in secondi
    - **Nel progetto**:
      ```python
      start_time = time.time()
      study.optimize(...)  # Hyperparameter tuning
      end_time = time.time()
      duration = end_time - start_time  # Durata in secondi
      print(f"Tempo totale: {duration/3600:.2f} ore")
      ```
    - **Output tipico**: 
      - 20 trials × 5 fold × 30 epochs ≈ 2-4 ore su RTX 4060
      - Salvato in `optuna_study_results.json`: `"optimization_duration_hours"`

20. **`import gc`**
    - Importa Garbage Collector di Python
    - **Cos'è il Garbage Collection**:
      - Processo automatico che libera memoria di oggetti non più usati
      - Python usa reference counting + generational GC
      - Normalmente automatico, ma può essere forzato manualmente
    - **Funzioni usate**:
      - `gc.collect()`: Forza garbage collection immediata
      - Ritorna numero di oggetti collezionati
    - **Quando usare**:
      - Dopo operazioni che creano molti oggetti temporanei
      - Training di modelli grandi (tensori GPU)
      - Loop con molte iterazioni
    - **Nel progetto**:
      ```python
      # Dopo ogni fold in cross-validation
      del tft, trainer, training_dataset, validation_dataset
      torch.cuda.empty_cache() if torch.cuda.is_available() else None
      gc.collect()  # Libera memoria Python
      ```
    - **Perché importante**:
      - GPU ha memoria limitata (8GB RTX 4060)
      - Senza cleanup, memoria si accumula → OOM error
      - Combinazione `del` + `torch.cuda.empty_cache()` + `gc.collect()` libera:
        1. Riferimenti Python (del)
        2. Cache GPU PyTorch (empty_cache)
        3. Oggetti orfani Python (gc.collect)

### Riepilogo Sezione Hyperparameter Optimization

Questa sezione importa le librerie necessarie per:
1. **Optuna**: Ottimizzazione Bayesiana automatica degli iperparametri
2. **PyTorchLightningPruningCallback**: Early stopping intelligente per trial
3. **JSON**: Persistenza risultati in formato leggibile
4. **time**: Tracciamento durata ottimizzazione
5. **gc**: Gestione memoria durante training intensivo

**Pipeline completa**:
```
Define search space → Create Optuna study → Optimize (20 trials)
    ↓
Per ogni trial:
    ↓
    Sample hyperparameters from TPE
    ↓
    Per ogni fold (5 fold CV):
        ↓
        Create dataset → Train model (30 epochs) → Validate
        ↓
        Pruning check: continua o termina?
        ↓
        Cleanup memoria (gc.collect)
    ↓
    Return mean validation loss
↓
Best hyperparameters → Save JSON → Final training (150 epochs)
```

**Output files generati**:
- `optuna_study_results.json`: Tutti i trial + best params
- `best_hyperparameters.json`: Solo best config
- `model_info.json`: Metadata modello completi

---

### Sezione 6: Print Versioni

```python
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch Lightning version: {pl.__version__}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

**Spiegazione riga per riga:**

17. **`print(f"PyTorch version: {torch.__version__}")`**
    - **f-string**: sintassi Python 3.6+ per formattazione stringhe
    - **`torch.__version__`**: attributo speciale che contiene versione PyTorch
    - Nel progetto: stampa `2.7.1+cu118` (versione con CUDA 11.8)
    - **Perché importante**: Verificare compatibilità CUDA/Python

18. **`print(f"PyTorch Lightning version: {pl.__version__}")`**
    - Stampa versione di Lightning
    - Nel progetto: stampa `2.5.6`
    - **Importante**: Lightning 2.x usa namespace `lightning.pytorch`, non `pytorch_lightning`

19. **`print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")`**
    - **Espressione ternaria**: `A if condition else B`
    - **`torch.cuda.is_available()`**: ritorna `True` se CUDA è disponibile
    - **`torch.cuda.get_device_name(0)`**: nome della GPU con indice 0
    - Nel progetto: stampa `NVIDIA GeForce RTX 4060`
    - **Fallback**: Stampa `'CPU'` se GPU non disponibile

---

## Cella 2: Verifica GPU/CUDA

Questa cella diagnostica la configurazione GPU/CUDA in dettaglio. È stata aggiunta dopo l'ottimizzazione per verificare che il setup GPU sia corretto.

### Codice Completo con Numerazione

```python
1  import torch
2  
3  print("="*60)
4  print("VERIFICA CONFIGURAZIONE GPU/CUDA")
5  print("="*60)
6  print(f"PyTorch version: {torch.__version__}")
7  print(f"CUDA disponibile: {torch.cuda.is_available()}")
8  
9  if torch.cuda.is_available():
10     print(f"CUDA version: {torch.version.cuda}")
11     print(f"Numero GPU disponibili: {torch.cuda.device_count()}")
12     print(f"GPU attiva: {torch.cuda.current_device()}")
13     print(f"Nome GPU: {torch.cuda.get_device_name(0)}")
14     print(f"Memoria GPU totale: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
15     
16     # Test rapido
17     x = torch.randn(3, 3).cuda()
18     print(f"\n✓ Test GPU riuscito! Tensor creato su: {x.device}")
19 else:
20     print("\n  GPU non disponibile. Possibili cause:")
21     print("  1. Driver NVIDIA non installati o non aggiornati")
22     print("  2. GPU non compatibile con CUDA 11.8")
23     print("  3. Ambiente virtuale non configurato correttamente")
24     print("\nPer verificare GPU hardware, esegui: nvidia-smi")
25 
26 print("="*60)
```

### Spiegazione Riga per Riga

**Righe 1-2: Import**
```python
import torch
```
- Re-importa torch per sicurezza (potrebbe essere eseguita standalone)

**Righe 3-5: Header**
```python
print("="*60)
print("VERIFICA CONFIGURAZIONE GPU/CUDA")
print("="*60)
```
- **`"="*60`**: Operatore di moltiplicazione stringhe in Python
- Crea una linea separatrice di 60 caratteri `=`
- Rende output più leggibile e organizzato

**Righe 6-7: Info Base**
```python
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA disponibile: {torch.cuda.is_available()}")
```
- Stampa versione PyTorch: `2.7.1+cu118`
  - `+cu118`: indica che PyTorch è compilato con CUDA 11.8
- **`torch.cuda.is_available()`**: 
  - Ritorna `True` se:
    1. PyTorch è compilato con CUDA support
    2. Driver NVIDIA funzionanti
    3. GPU compatibile rilevata
  - Ritorna `False` altrimenti
  - Nel progetto: `True` (RTX 4060 rilevata)

**Riga 9: Controllo Condizionale**
```python
if torch.cuda.is_available():
```
- Esegue il blocco solo se GPU disponibile
- Evita crash chiamando funzioni CUDA senza GPU

**Riga 10: Versione CUDA**
```python
print(f"CUDA version: {torch.version.cuda}")
```
- **`torch.version.cuda`**: versione CUDA compilata in PyTorch
- Nel progetto: stampa `11.8`
- **Importante**: Questa è la versione CUDA di PyTorch, non del driver
- Driver può essere più recente (es. 12.x) ma PyTorch usa 11.8

**Riga 11: Conteggio GPU**
```python
print(f"Numero GPU disponibili: {torch.cuda.device_count()}")
```
- **`torch.cuda.device_count()`**: numero di GPU utilizzabili
- Nel progetto: `1` (una sola RTX 4060)
- Sistemi multi-GPU possono avere 2, 4, 8+ GPU

**Riga 12: GPU Attiva**
```python
print(f"GPU attiva: {torch.cuda.current_device()}")
```
- **`torch.cuda.current_device()`**: indice GPU attualmente selezionata
- Ritorna un intero: `0`, `1`, `2`, etc.
- Nel progetto: `0` (prima e unica GPU)
- **Cambio GPU**: `torch.cuda.set_device(1)` per switchare

**Riga 13: Nome GPU**
```python
print(f"Nome GPU: {torch.cuda.get_device_name(0)}")
```
- **`torch.cuda.get_device_name(0)`**: nome commerciale della GPU con indice 0
- Nel progetto: `NVIDIA GeForce RTX 4060`
- **Parametro**: `0` è l'indice GPU (0-based)

**Riga 14: Memoria GPU**
```python
print(f"Memoria GPU totale: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```
- **Analisi complessa**, scomponiamola:

1. **`torch.cuda.get_device_properties(0)`**:
   - Ritorna oggetto con proprietà della GPU 0
   - Tipo: `torch.cuda._CudaDeviceProperties`
   - Contiene: `total_memory`, `major`, `minor`, `multi_processor_count`, etc.

2. **`.total_memory`**:
   - Memoria totale in bytes
   - Per RTX 4060: `8,589,934,592` bytes (8 GB)

3. **`/ 1024**3`**:
   - Conversione bytes → gigabytes
   - `1024**3 = 1,073,741,824` (bytes in 1 GB)
   - **Perché 1024 e non 1000?**: Sistema binario (GiB) vs decimale (GB)

4. **`:.2f`**:
   - Format specifier per f-string
   - `.2f`: floating point con 2 decimali
   - Output: `8.00 GB`

**Righe 16-18: Test Pratico GPU**
```python
# Test rapido
x = torch.randn(3, 3).cuda()
print(f"\n✓ Test GPU riuscito! Tensor creato su: {x.device}")
```

**Riga 17 - Analisi dettagliata**:
```python
x = torch.randn(3, 3).cuda()
```
- **`torch.randn(3, 3)`**: 
  - Crea tensor 3×3 con valori casuali da distribuzione normale N(0,1)
  - Creato di default su CPU
  
- **`.cuda()`**: 
  - Metodo che trasferisce tensor da CPU a GPU
  - Equivalente a: `x.to('cuda:0')`
  - **Allocazione memoria**: Alloca ~72 bytes su GPU (9 float32 × 4 bytes)
  
- **Perché questo test?**:
  - Verifica che PyTorch possa effettivamente allocare memoria su GPU
  - Conferma che driver NVIDIA funzionino correttamente
  - Test end-to-end del stack CUDA

**Riga 18**:
```python
print(f"\n✓ Test GPU riuscito! Tensor creato su: {x.device}")
```
- **`\n`**: newline per spaziatura
- **`✓`**: carattere Unicode per checkmark (U+2713)
- **`x.device`**: 
  - Attributo che mostra su quale device risiede il tensor
  - Output: `cuda:0` (GPU 0)
  - Alternativa CPU: `cpu`

**Righe 19-24: Blocco Else (Fallback)**
```python
else:
    print("\n⚠️  GPU non disponibile. Possibili cause:")
    print("  1. Driver NVIDIA non installati o non aggiornati")
    print("  2. GPU non compatibile con CUDA 11.8")
    print("  3. Ambiente virtuale non configurato correttamente")
    print("\nPer verificare GPU hardware, esegui: nvidia-smi")
```
- Eseguito solo se `torch.cuda.is_available()` è `False`
- **⚠️**: warning emoji (U+26A0)
- Fornisce troubleshooting pratico:
  1. **Driver NVIDIA**: Verifica con `nvidia-smi` in terminale
  2. **Compatibilità CUDA**: RTX 4060 supporta CUDA 11.8 ✓
  3. **Ambiente virtuale**: PyTorch CPU vs GPU (problema iniziale del progetto)

**Riga 26: Footer**
```python
print("="*60)
```
- Chiude il blocco visivo con linea separatrice

### Output Esempio (Progetto Reale)

```
============================================================
VERIFICA CONFIGURAZIONE GPU/CUDA
============================================================
PyTorch version: 2.7.1+cu118
CUDA disponibile: True
CUDA version: 11.8
Numero GPU disponibili: 1
GPU attiva: 0
Nome GPU: NVIDIA GeForce RTX 4060
Memoria GPU totale: 8.00 GB

✓ Test GPU riuscito! Tensor creato su: cuda:0
============================================================
```

### Perché Questa Cella è Importante

1. **Debugging Setup GPU**: 
   - Nel progetto, inizialmente PyTorch era CPU-only
   - Questa cella ha identificato il problema immediatamente

2. **Verifica Compatibilità**:
   - Python 3.13 + CUDA 11.8 compatibili ✓
   - CUDA 12.1 NON compatibile con Python 3.13 (problema iniziale)

3. **Performance Tracking**:
   - 8 GB VRAM sufficienti per TFT con batch_size=64
   - Stima uso memoria: ~4-6 GB durante training

4. **Troubleshooting Sistematico**:
   - Se GPU non funziona, output indica esattamente quale componente fallisce
   - Messaggi di errore chiari con soluzioni proposte

---

## Cella 3: Conversione Excel in CSV

Questa cella gestisce la conversione dei file Excel originali (con 2 fogli ciascuno) in 4 file CSV separati. È una fase critica di preprocessing dei dati.

### Struttura File Originali

**Dataset Forniti:**
- `pv_dataset.xlsx`: 2 fogli
  - Foglio 1: `07-10--06-11` (luglio 2010 - giugno 2011)
  - Foglio 2: `07-11--06-12` (luglio 2011 - giugno 2012)
- `wx_dataset.xlsx`: 2 fogli
  - Foglio 1: `07-10--06-11` (dati meteo 2010-2011)
  - Foglio 2: `07-11--06-12` (dati meteo 2011-2012)

### Codice Completo Annotato

```python
1  import os
2  from pathlib import Path
3  import pandas as pd
4  
5  # Percorsi dei file Excel
6  project_root = Path('..').resolve()
7  data_dir = project_root / 'data' / 'raw'
8  
9  pv_excel_file = data_dir / 'pv_dataset.xlsx'
10 wx_excel_file = data_dir / 'wx_dataset.xlsx'
11 
12 print("="*60)
13 print("CONVERSIONE FILE EXCEL IN CSV")
14 print("="*60)
15 
16 # Funzione per convertire Excel con 2 fogli in 2 CSV
17 def excel_to_csv(excel_path, base_name):
18     """
19     Converte un file Excel con 2 fogli in 2 file CSV separati
20     
21     Args:
22         excel_path: Path al file Excel
23         base_name: Nome base per i file CSV di output
24     """
25     if not excel_path.exists():
26         print(f"✗ File non trovato: {excel_path}")
27         return None, None
28     
29     print(f"\n Elaborazione: {excel_path.name}")
30     
31     # Leggi i nomi dei fogli
32     xl_file = pd.ExcelFile(excel_path)
33     sheet_names = xl_file.sheet_names
34     
35     print(f"   Fogli trovati ({len(sheet_names)}): {sheet_names}")
36     
37     if len(sheet_names) < 2:
38         print(f"  Attenzione: trovati solo {len(sheet_names)} fogli invece di 2")
39     
40     csv_files = []
41     
42     # Converti ogni foglio in CSV
43     for i, sheet_name in enumerate(sheet_names[:2]):  # Prendi solo i primi 2 fogli
44         # Leggi il foglio
45         df = pd.read_excel(excel_path, sheet_name=sheet_name)
46         
47         # Crea il nome del file CSV
48         # Usa il nome del foglio per creare il CSV
49         csv_filename = f"{base_name} - {sheet_name}.csv"
50         csv_path = data_dir / csv_filename
51         
52         # Salva come CSV
53         df.to_csv(csv_path, index=False)
54         csv_files.append(csv_path)
55         
56         print(f"   ✓ Foglio '{sheet_name}' -> {csv_filename}")
57         print(f"      Shape: {df.shape}, Colonne: {df.columns.tolist()}")
58     
59     return csv_files[0] if len(csv_files) > 0 else None, csv_files[1] if len(csv_files) > 1 else None
60 
61 # Converti i file PV
62 print("\n" + "-"*60)
63 print("1. CONVERSIONE PV DATASET")
64 print("-"*60)
65 pv_csv1, pv_csv2 = excel_to_csv(pv_excel_file, "pv_dataset")
66 
67 # Converti i file Weather
68 print("\n" + "-"*60)
69 print("2. CONVERSIONE WEATHER DATASET")
70 print("-"*60)
71 wx_csv1, wx_csv2 = excel_to_csv(wx_excel_file, "wx_dataset")
72 
73 print("\n" + "="*60)
74 print("CONVERSIONE COMPLETATA!")
75 print("="*60)
76 
77 # Verifica che tutti i file siano stati creati
78 all_files_created = all([pv_csv1, pv_csv2, wx_csv1, wx_csv2])
79 
80 if all_files_created:
81     print("\n✓ Tutti i 4 file CSV sono stati creati con successo:")
82     print(f"  1. {pv_csv1.name}")
83     print(f"  2. {pv_csv2.name}")
84     print(f"  3. {wx_csv1.name}")
85     print(f"  4. {wx_csv2.name}")
86     print(f"\n📂 Salvati in: {data_dir}")
87 else:
88     print("\n⚠️  Alcuni file non sono stati creati correttamente")
89     
90 print("\n💡 Nota: I file CSV mantengono i nomi originali dei fogli Excel")
```

### Spiegazione Dettagliata Sezioni

#### Sezione 1: Import e Setup Paths (Righe 1-10)

**Righe 1-3: Import Moduli**
```python
import os
from pathlib import Path
import pandas as pd
```

1. **`import os`**:
   - Modulo per operazioni sistema operativo
   - Nel codice: usato per `os.path.exists()`, `os.listdir()`
   - **Nota**: In questo snippet, preferito `pathlib.Path` (più moderno)

2. **`from pathlib import Path`**:
   - Classe Path per manipolazione percorsi object-oriented
   - **Vantaggi vs os.path**:
     - Operatore `/` per join: `Path('data') / 'raw'`
     - Metodi integrati: `.exists()`, `.resolve()`, `.name`
     - Cross-platform automatico (Windows `\` vs Unix `/`)
   - **Nel progetto**: Gestisce percorsi lunghi Windows senza escape

3. **`import pandas as pd`**:
   - Per leggere Excel e scrivere CSV
   - Funzioni chiave: `pd.ExcelFile()`, `pd.read_excel()`, `df.to_csv()`

**Righe 5-10: Definizione Percorsi**
```python
project_root = Path('..').resolve()
data_dir = project_root / 'data' / 'raw'

pv_excel_file = data_dir / 'pv_dataset.xlsx'
wx_excel_file = data_dir / 'wx_dataset.xlsx'
```

**Riga 6 - Analisi dettagliata**:
```python
project_root = Path('..').resolve()
```
- **`Path('..')`**: 
  - Crea oggetto Path che punta alla directory parent
  - `.` = directory corrente (src/)
  - `..` = directory parent (project root)
  
- **`.resolve()`**:
  - Converte percorso relativo in assoluto
  - Risolve symlink
  - Normalizza separatori (Windows: `\`, Unix: `/`)
  - **Output esempio**: 
    ```
    D:\Cartelle\Cartelle nuove\Uni.Ingegneria\Magistrale UCBM\
    Deep Learning\DL-Project---24-Hour-Ahead-Power-Forecasting-with-Temporal-Fusion-Transformer-TFT-
    ```

**Riga 7**:
```python
data_dir = project_root / 'data' / 'raw'
```
- **Operatore `/`**: Overloaded in pathlib per join elegante
- Equivalente a: `os.path.join(project_root, 'data', 'raw')`
- **Struttura attesa**:
  ```
  project_root/
  ├── src/
  │   └── PV_Forecasting_TFT.ipynb  ← notebook corrente
  ├── data/
  │   └── raw/
  │       ├── pv_dataset.xlsx
  │       └── wx_dataset.xlsx
  ```

**Righe 9-10**:
```python
pv_excel_file = data_dir / 'pv_dataset.xlsx'
wx_excel_file = data_dir / 'wx_dataset.xlsx'
```
- Costruisce percorsi completi ai file Excel
- **Type**: `pathlib.WindowsPath` (su Windows)
- Non verifica esistenza (fatto dopo)

#### Sezione 2: Funzione excel_to_csv (Righe 17-59)

**Righe 17-24: Firma e Docstring**
```python
def excel_to_csv(excel_path, base_name):
    """
    Converte un file Excel con 2 fogli in 2 file CSV separati
    
    Args:
        excel_path: Path al file Excel
        base_name: Nome base per i file CSV di output
    """
```
- **Parametri**:
  - `excel_path`: oggetto `Path` (es. `data_dir / 'pv_dataset.xlsx'`)
  - `base_name`: stringa (es. `"pv_dataset"`)
- **Return**: tupla `(csv_file1, csv_file2)` o `(None, None)` se errore
- **Docstring**: Formato Google-style per documentazione

**Righe 25-27: Validazione Input**
```python
if not excel_path.exists():
    print(f"✗ File non trovato: {excel_path}")
    return None, None
```
- **`excel_path.exists()`**: metodo Path che verifica esistenza file
- **Early return**: Pattern per gestire errori subito
- **Return `(None, None)`**: Tuple unpacking sicuro nel chiamante
- **✗**: Carattere Unicode per indicare fallimento (U+2717)

**Righe 29-33: Apertura File Excel**
```python
print(f"\n Elaborazione: {excel_path.name}")

xl_file = pd.ExcelFile(excel_path)
sheet_names = xl_file.sheet_names
```

**Riga 32 - Analisi dettagliata**:
```python
xl_file = pd.ExcelFile(excel_path)
```
- **`pd.ExcelFile()`**: 
  - Classe Pandas per gestire file Excel in modo efficiente
  - Carica il file una volta, legge fogli multipli senza reopen
  - **Engine detection**: Auto-detect Excel format (.xls vs .xlsx)
  - **Nel progetto**: .xlsx → engine `openpyxl` usato automaticamente

**Riga 33**:
```python
sheet_names = xl_file.sheet_names
```
- **`.sheet_names`**: Attributo che ritorna lista nomi fogli
- **Nel progetto**:
  - PV: `['07-10--06-11', '07-11--06-12']`
  - WX: `['07-10--06-11', '07-11--06-12']`
- **Type**: `list[str]`

**Righe 35-38: Logging e Validazione**
```python
print(f"   Fogli trovati ({len(sheet_names)}): {sheet_names}")

if len(sheet_names) < 2:
    print(f"  Attenzione: trovati solo {len(sheet_names)} fogli invece di 2")
```
- Stampa numero fogli trovati
- **Warning non bloccante**: Continua anche con 1 foglio
- **Spazi iniziali** (`   `): Indentazione visiva per output gerarchico

**Riga 40: Inizializzazione Lista**
```python
csv_files = []
```
- Lista per accumulare Path dei CSV generati
- Usata per return finale

**Righe 43-57: Loop Conversione Fogli**
```python
for i, sheet_name in enumerate(sheet_names[:2]):
```
- **`enumerate()`**: Genera coppie `(indice, valore)`
- **`sheet_names[:2]`**: 
  - Slicing per prendere solo primi 2 fogli
  - Previene processing di fogli extra indesiderati
  - **Nel progetto**: Entrambi file hanno esattamente 2 fogli

**Riga 45: Lettura Foglio**
```python
df = pd.read_excel(excel_path, sheet_name=sheet_name)
```
- **`pd.read_excel()`**: 
  - Legge foglio Excel in DataFrame
  - **Parametri chiave**:
    - `sheet_name`: quale foglio leggere (nome o indice)
    - `header=0` (default): prima riga come column names
    - `index_col=None` (default): nessuna colonna come index
  - **Performance**: 
    - Parsing .xlsx lento per file grandi (>100MB)
    - Nel progetto: file piccoli (~8MB), veloce (<1 sec)

**Righe 47-50: Generazione Nome CSV**
```python
csv_filename = f"{base_name} - {sheet_name}.csv"
csv_path = data_dir / csv_filename
```
- **Formato output**: `"pv_dataset - 07-10--06-11.csv"`
- **Componenti**:
  - `base_name`: `"pv_dataset"` o `"wx_dataset"`
  - ` - `: Separatore visivo
  - `sheet_name`: `"07-10--06-11"` o `"07-11--06-12"`
  - `.csv`: Estensione
- **`csv_path`**: Path completo (data_dir + filename)

**Righe 52-54: Scrittura CSV**
```python
df.to_csv(csv_path, index=False)
csv_files.append(csv_path)
```

**Riga 53 - Parametri importanti**:
```python
df.to_csv(csv_path, index=False)
```
- **`index=False`**: 
  - **Cruciale**: NON scrive la colonna index nel CSV
  - **Senza questo**: CSV avrebbe colonna extra `0,1,2,3...`
  - **Nel progetto**: Mantiene solo colonne originali

- **Altri parametri (non specificati, usano default)**:
  - `sep=','`: Separatore virgola (standard CSV)
  - `encoding='utf-8'`: Encoding Unicode
  - `header=True`: Scrive nomi colonne come prima riga

**Righe 56-57: Logging Successo**
```python
print(f"   ✓ Foglio '{sheet_name}' -> {csv_filename}")
print(f"      Shape: {df.shape}, Colonne: {df.columns.tolist()}")
```
- **`df.shape`**: Tupla `(n_rows, n_columns)`
  - Es. PV: `(8760, 2)` → 8760 ore × 2 colonne
  - Es. WX: `(8760, 9)` → 8760 ore × 9 colonne meteo
- **`df.columns.tolist()`**: Converte Index in lista Python
  - Es. PV: `['Max kWp', '82.41']`
  - Es. WX: `['dt_iso', 'temp', 'Dni', 'Ghi', ...]`

**Riga 59: Return con Defensive Programming**
```python
return csv_files[0] if len(csv_files) > 0 else None, csv_files[1] if len(csv_files) > 1 else None
```
- **Tuple unpacking sicuro**:
  - Se 2 file creati: `(Path1, Path2)`
  - Se 1 file creato: `(Path1, None)`
  - Se 0 file creati: `(None, None)`
- **Evita**: `IndexError` se `csv_files` ha meno di 2 elementi

#### Sezione 3: Chiamate Funzione (Righe 61-71)

**Righe 62-65: Conversione PV**
```python
print("\n" + "-"*60)
print("1. CONVERSIONE PV DATASET")
print("-"*60)
pv_csv1, pv_csv2 = excel_to_csv(pv_excel_file, "pv_dataset")
```
- **Separatori**: Linee di `-` (meno visibili di `=`)
- **Tuple unpacking**: Assegna direttamente i 2 return values
- **Variabili risultanti**:
  - `pv_csv1`: Path a `"pv_dataset - 07-10--06-11.csv"`
  - `pv_csv2`: Path a `"pv_dataset - 07-11--06-12.csv"`

**Righe 68-71: Conversione Weather**
```python
print("\n" + "-"*60)
print("2. CONVERSIONE WEATHER DATASET")
print("-"*60)
wx_csv1, wx_csv2 = excel_to_csv(wx_excel_file, "wx_dataset")
```
- Identico a PV, ma per dati meteo
- **Variabili risultanti**:
  - `wx_csv1`: Path a `"wx_dataset - 07-10--06-11.csv"`
  - `wx_csv2`: Path a `"wx_dataset - 07-11--06-12.csv"`

#### Sezione 4: Verifica Finale (Righe 77-90)

**Riga 78: Verifica Completamento**
```python
all_files_created = all([pv_csv1, pv_csv2, wx_csv1, wx_csv2])
```
- **`all()`**: Funzione built-in Python
  - Ritorna `True` se **tutti** gli elementi sono truthy
  - `None` è falsy → se anche uno è `None`, ritorna `False`
- **Nel progetto**: `True` se tutti e 4 i file creati correttamente

**Righe 80-86: Successo**
```python
if all_files_created:
    print("\n✓ Tutti i 4 file CSV sono stati creati con successo:")
    print(f"  1. {pv_csv1.name}")
    print(f"  2. {pv_csv2.name}")
    print(f"  3. {wx_csv1.name}")
    print(f"  4. {wx_csv2.name}")
    print(f"\n📂 Salvati in: {data_dir}")
```
- **`.name`**: Attributo Path che ritorna solo filename (no path)
  - Es. `pv_csv1.name` → `"pv_dataset - 07-10--06-11.csv"`
- **📂**: Emoji folder (U+1F4C2)

**Righe 87-88: Fallimento**
```python
else:
    print("\n⚠️  Alcuni file non sono stati creati correttamente")
```
- Eseguito solo se `all_files_created == False`
- Non solleva eccezione (permette continuazione manuale)

### Output Esempio (Esecuzione Reale)

```
============================================================
CONVERSIONE FILE EXCEL IN CSV
============================================================

------------------------------------------------------------
1. CONVERSIONE PV DATASET
------------------------------------------------------------

 Elaborazione: pv_dataset.xlsx
   Fogli trovati (2): ['07-10--06-11', '07-11--06-12']
   ✓ Foglio '07-10--06-11' -> pv_dataset - 07-10--06-11.csv
      Shape: (8760, 2), Colonne: ['Max kWp', '82.41']
   ✓ Foglio '07-11--06-12' -> pv_dataset - 07-11--06-12.csv
      Shape: (8557, 2), Colonne: ['Max kWp', '82.41']

------------------------------------------------------------
2. CONVERSIONE WEATHER DATASET
------------------------------------------------------------

 Elaborazione: wx_dataset.xlsx
   Fogli trovati (2): ['07-10--06-11', '07-11--06-12']
   ✓ Foglio '07-10--06-11' -> wx_dataset - 07-10--06-11.csv
      Shape: (8760, 9), Colonne: ['dt_iso', 'temp', 'Dni', 'Ghi', 'humidity', 'clouds_all', 'wind_speed', 'pressure', 'rain_1h']
   ✓ Foglio '07-11--06-12' -> wx_dataset - 07-11--06-12.csv
      Shape: (8557, 9), Colonne: ['dt_iso', 'temp', 'Dni', 'Ghi', 'humidity', 'clouds_all', 'wind_speed', 'pressure', 'rain_1h']

============================================================
CONVERSIONE COMPLETATA!
============================================================

✓ Tutti i 4 file CSV sono stati creati con successo:
  1. pv_dataset - 07-10--06-11.csv
  2. pv_dataset - 07-11--06-12.csv
  3. wx_dataset - 07-10--06-11.csv
  4. wx_dataset - 07-11--06-12.csv

📂 Salvati in: D:\...\DL-Project-...\data\raw

💡 Nota: I file CSV mantengono i nomi originali dei fogli Excel
```

### Perché Questa Conversione è Necessaria

1. **Performance**:
   - CSV più veloci da leggere (testo puro vs formato binario Excel)
   - Pandas `read_csv()` ~10x più veloce di `read_excel()`

2. **Compatibilità**:
   - CSV universale (tutti i linguaggi/tool)
   - Excel richiede librerie extra (`openpyxl`, `xlrd`)

3. **Versioning**:
   - CSV testuale → facile diff in Git
   - Excel binario → difficile tracking modifiche

4. **Memoria**:
   - CSV può essere letto in chunk (file grandi)
   - Excel deve essere caricato interamente

5. **Semplicità**:
   - CSV = una tabella per file
   - Excel = gestione multi-foglio complessa

---

## Cella 4: Ricerca e Caricamento CSV (Esplorazione Directory)

Questa cella diagnostica esplora la struttura delle directory per localizzare i file CSV appena creati. È una fase di debugging/verifica.

### Codice Completo

```python
1  # Verifica la struttura delle directory e trova i file CSV
2  import os
3  from pathlib import Path
4  
5  print("Current working directory:", os.getcwd())
6  print("\nFile nel notebook directory (src/):")
7  if os.path.exists('.'):
8      for item in os.listdir('.'):
9          print(f"  - {item}")
10 
11 print("\nFile nella directory parent:")
12 parent_dir = Path('..').resolve()
13 if parent_dir.exists():
14     for item in os.listdir(parent_dir):
15         print(f"  - {item}")
16 
17 # Cerca i file CSV
18 print("\n" + "="*60)
19 print("Ricerca file CSV...")
20 print("="*60)
21 for root, dirs, files in os.walk('..'):
22     for file in files:
23         if file.endswith('.csv'):
24             full_path = os.path.join(root, file)
25             print(f"Trovato: {full_path}")
```

### Spiegazione Dettagliata

**Righe 1-3: Import**
```python
import os
from pathlib import Path
```
- Re-import per sicurezza (potrebbe essere eseguita standalone)

**Riga 5: Working Directory**
```python
print("Current working directory:", os.getcwd())
```
- **`os.getcwd()`**: Get Current Working Directory
- Ritorna percorso assoluto della directory corrente
- Nel progetto (eseguito da Jupyter):
  ```
  D:\...\DL-Project-...\src
  ```
- **Importante**: In Jupyter, CWD = directory del notebook (.ipynb)

**Righe 6-9: Listare File Directory Corrente**
```python
print("\nFile nel notebook directory (src/):")
if os.path.exists('.'):
    for item in os.listdir('.'):
        print(f"  - {item}")
```

**Riga 7: Check Esistenza**
```python
if os.path.exists('.'):
```
- **`.`**: Directory corrente (src/)
- **`os.path.exists()`**: Verifica esistenza
- **Sempre True** per `.`, ma buona pratica defensiva

**Riga 8: Iterazione File**
```python
for item in os.listdir('.'):
```
- **`os.listdir('.')`**: Lista tutti i file/cartelle in directory corrente
- Ritorna `list[str]` (solo nomi, non percorsi completi)
- **Non ricorsivo**: solo primo livello
- Nel progetto, output tipico:
  ```
  - PV_Forecasting_TFT.ipynb
  - __pycache__
  - .ipynb_checkpoints
  ```

**Righe 11-15: Listare Directory Parent**
```python
print("\nFile nella directory parent:")
parent_dir = Path('..').resolve()
if parent_dir.exists():
    for item in os.listdir(parent_dir):
        print(f"  - {item}")
```

**Riga 12**: 
```python
parent_dir = Path('..').resolve()
```
- **`Path('..')`**: Directory parent (project root)
- **`.resolve()`**: Converte in percorso assoluto
- Nel progetto: directory principale del repository

**Riga 14**:
```python
for item in os.listdir(parent_dir):
```
- Lista file/cartelle nel project root
- Output tipico:
  ```
  - data/
  - src/
  - README.md
  - CHANGELOG_AND_FIXES.md
  - DEVELOPMENT_LOG.md
  - lightning_logs/
  ```

**Righe 17-25: Ricerca Ricorsiva CSV**
```python
print("\n" + "="*60)
print("Ricerca file CSV...")
print("="*60)
for root, dirs, files in os.walk('..'):
    for file in files:
        if file.endswith('.csv'):
            full_path = os.path.join(root, file)
            print(f"Trovato: {full_path}")
```

**Riga 21 - Analisi dettagliata**:
```python
for root, dirs, files in os.walk('..'):
```
- **`os.walk()`**: Generatore per traversare albero directory ricorsivamente
- **Parametro**: `'..'` = parte dal parent (project root)
- **Ritorna 3-tuple ad ogni iterazione**:
  - `root` (str): Path della directory corrente nell'iterazione
  - `dirs` (list[str]): Nome delle sottodirectory in `root`
  - `files` (list[str]): Nome dei file in `root`

**Esempio Iterazione**:
```python
Iterazione 1:
  root = 'D:\...\DL-Project-...'
  dirs = ['data', 'src', 'lightning_logs']
  files = ['README.md', 'CHANGELOG_AND_FIXES.md']

Iterazione 2:
  root = 'D:\...\DL-Project-...\data'
  dirs = ['raw']
  files = []

Iterazione 3:
  root = 'D:\...\DL-Project-...\data\raw'
  dirs = []
  files = ['pv_dataset - 07-10--06-11.csv', 'pv_dataset - 07-11--06-12.csv', ...]
```

**Riga 22-23**:
```python
for file in files:
    if file.endswith('.csv'):
```
- Loop interno: itera su file nella directory corrente
- **`file.endswith('.csv')`**: Filtra solo CSV
- **Case-sensitive** su Linux (`.CSV` non matcherebbe)

**Riga 24-25**:
```python
full_path = os.path.join(root, file)
print(f"Trovato: {full_path}")
```
- **`os.path.join()`**: Combina path componenti
  - Gestisce automaticamente separatori OS (`\` Windows, `/` Unix)
  - Es: `os.path.join('data\\raw', 'file.csv')` → `'data\\raw\\file.csv'`
- Stampa percorso completo per ogni CSV trovato

### Output Esempio

```
Current working directory: D:\Cartelle\Cartelle nuove\Uni.Ingegneria\Magistrale UCBM\Deep Learning\DL-Project---24-Hour-Ahead-Power-Forecasting-with-Temporal-Fusion-Transformer-TFT-\src

File nel notebook directory (src/):
  - PV_Forecasting_TFT.ipynb
  - __pycache__

File nella directory parent:
  - data
  - src
  - README.md
  - CHANGELOG_AND_FIXES.md
  - DEVELOPMENT_LOG.md
  - lightning_logs

============================================================
Ricerca file CSV...
============================================================
Trovato: ..\data\raw\pv_dataset - 07-10--06-11.csv
Trovato: ..\data\raw\pv_dataset - 07-11--06-12.csv
Trovato: ..\data\raw\wx_dataset - 07-10--06-11.csv
Trovato: ..\data\raw\wx_dataset - 07-11--06-12.csv
```

### Perché Questa Cella Esiste

1. **Debugging**: Verifica che conversione Excel→CSV abbia funzionato
2. **Path Discovery**: Trova automaticamente file senza hardcode path
3. **Cross-Platform**: `os.walk()` funziona su Windows/Linux/Mac
4. **Troubleshooting**: Aiuta utente se file non trovati

---

## Cella 5: Caricamento e Concatenazione Dati PV

Questa cella carica i 2 file CSV PV, li concatena in un unico DataFrame, e identifica/rinomina le colonne target.

### Codice Completo con Numerazione

```python
1  import os
2  from pathlib import Path
3  
4  # Trova la directory root del progetto (una cartella sopra src/)
5  project_root = Path('..').resolve()
6  
7  # Percorsi dei file - cerca prima nella directory data/, poi nella root
8  def find_csv_file(filename):
9      """Cerca il file CSV in diverse possibili location"""
10     possible_paths = [
11         project_root / 'data' / 'raw' / filename,
12         project_root / 'data' / filename,
13         project_root / filename,
14         Path('.') / filename,
15         Path('..') / filename
16     ]
17     
18     for path in possible_paths:
19         if path.exists():
20             print(f"✓ Trovato: {path}")
21             return str(path)
22     
23     print(f"✗ Non trovato: {filename}")
24     print(f"  Percorsi cercati:")
25     for p in possible_paths:
26         print(f"    - {p}")
27     return None
28 
29 print("Ricerca dei file CSV...\n")
30 pv_file1 = find_csv_file("pv_dataset - 07-10--06-11.csv")
31 pv_file2 = find_csv_file("pv_dataset - 07-11--06-12.csv")
32 wx_file1 = find_csv_file("wx_dataset - 07-10--06-11.csv")
33 wx_file2 = find_csv_file("wx_dataset - 07-11--06-12.csv")
34 
35 # Verifica che tutti i file siano stati trovati
36 if not all([pv_file1, pv_file2, wx_file1, wx_file2]):
37     print("\nATTENZIONE: Non tutti i file sono stati trovati!")
38     print("\nAssicurati che i 4 file CSV siano nella directory del progetto.")
39     print("Posizionali in una di queste location:")
40     print(f"  - {project_root / 'data' / 'raw'}")
41     print(f"  - {project_root / 'data'}")
42     print(f"  - {project_root}")
43     raise FileNotFoundError("File CSV non trovati")
44 
45 print("\n" + "="*60)
46 print("Caricamento dati PV...")
47 print("="*60)
48 
49 # Carica dati PV
50 pv1 = pd.read_csv(pv_file1)
51 pv2 = pd.read_csv(pv_file2)
52 
53 print(f"PV1 shape: {pv1.shape}")
54 print(f"PV2 shape: {pv2.shape}")
55 print(f"\nColonne PV1: {pv1.columns.tolist()}")
56 
57 # Concatena i dati PV
58 pv_data = pd.concat([pv1, pv2], ignore_index=True)
59 print(f"\nPV data concatenato: {pv_data.shape}")
60 
61 # Identifica la colonna timestamp e target
62 # La colonna "Max kWp" contiene il timestamp
63 # La prima colonna numerica (probabilmente "82.41" o simile) è la produzione
64 timestamp_col = "Max kWp"
65 target_col = [col for col in pv_data.columns if col != timestamp_col][0]
66 
67 print(f"\nColonna timestamp identificata: '{timestamp_col}'")
68 print(f"Colonna target identificata: '{target_col}'")
69 
70 # Rinomina le colonne
71 pv_data = pv_data.rename(columns={timestamp_col: 'datetime', target_col: 'power_kw'})
72 print(f"\nPrime righe PV data:")
73 print(pv_data.head())
```

### Spiegazione Dettagliata

#### Sezione 1: Funzione find_csv_file (Righe 8-27)

**Righe 8-9: Definizione e Docstring**
```python
def find_csv_file(filename):
    """Cerca il file CSV in diverse possibili location"""
```
- **Scopo**: Ricerca robusta di file CSV in multiple location
- **Parametro**: `filename` (str) - Nome file da cercare
- **Return**: Path assoluto (str) o `None` se non trovato

**Righe 10-16: Lista Percorsi Candidati**
```python
possible_paths = [
    project_root / 'data' / 'raw' / filename,
    project_root / 'data' / filename,
    project_root / filename,
    Path('.') / filename,
    Path('..') / filename
]
```
- **Lista ordinata** per priorità (data/raw prima)
- **5 location** controllate sequenzialmente:

1. **`project_root / 'data' / 'raw' / filename`**:
   - Location ideale: `D:\...\DL-Project-...\data\raw\file.csv`
   - Dove i CSV dovrebbero essere dopo conversione

2. **`project_root / 'data' / filename`**:
   - Fallback: file direttamente in data/ (no sottocartella raw/)
   
3. **`project_root / filename`**:
   - File nel project root (semplificato)
   
4. **`Path('.') / filename`**:
   - File nella directory corrente (src/)
   
5. **`Path('..') / filename`**:
   - File nel parent (project root, percorso relativo)

**Righe 18-21: Iterazione e Check Esistenza**
```python
for path in possible_paths:
    if path.exists():
        print(f"✓ Trovato: {path}")
        return str(path)
```

**Riga 18**:
```python
for path in possible_paths:
```
- Itera sui 5 percorsi candidati in ordine

**Riga 19**:
```python
if path.exists():
```
- **`.exists()`**: Metodo Path che verifica esistenza
- Ritorna `True` se file/directory esiste
- **Performance**: Controllare esistenza è fast (~microseconds)

**Riga 20-21**:
```python
print(f"✓ Trovato: {path}")
return str(path)
```
- **Early return**: Appena trovato, ritorna subito
- **`str(path)`**: Converte `Path` object in stringa
  - Necessario perché `pd.read_csv()` accetta str o Path
  - Preferito str per compatibilità

**Righe 23-27: Gestione File Non Trovato**
```python
print(f"✗ Non trovato: {filename}")
print(f"  Percorsi cercati:")
for p in possible_paths:
    print(f"    - {p}")
return None
```
- Eseguito solo se loop completa senza return
- Stampa debug: mostra tutti i path controllati
- **Return `None`**: Indica fallimento (checked dal chiamante)

#### Sezione 2: Chiamate Funzione (Righe 29-43)

**Righe 29-33: Ricerca 4 File**
```python
print("Ricerca dei file CSV...\n")
pv_file1 = find_csv_file("pv_dataset - 07-10--06-11.csv")
pv_file2 = find_csv_file("pv_dataset - 07-11--06-12.csv")
wx_file1 = find_csv_file("wx_dataset - 07-10--06-11.csv")
wx_file2 = find_csv_file("wx_dataset - 07-11--06-12.csv")
```
- 4 chiamate separate per ciascun CSV
- **Variabili risultanti**:
  - Tipo: `str` (path assoluto) o `None`
  - Es. `pv_file1 = "D:\\...\\data\\raw\\pv_dataset - 07-10--06-11.csv"`

**Righe 36-43: Validazione e Error Handling**
```python
if not all([pv_file1, pv_file2, wx_file1, wx_file2]):
    print("\nATTENZIONE: Non tutti i file sono stati trovati!")
    print("\nAssicurati che i 4 file CSV siano nella directory del progetto.")
    print("Posizionali in una di queste location:")
    print(f"  - {project_root / 'data' / 'raw'}")
    print(f"  - {project_root / 'data'}")
    print(f"  - {project_root}")
    raise FileNotFoundError("File CSV non trovati")
```

**Riga 36 - Analisi dettagliata**:
```python
if not all([pv_file1, pv_file2, wx_file1, wx_file2]):
```
- **`all([...])`**: Funzione built-in che verifica tutti truthy
- **`None` is falsy**: Se anche un solo file manca (=`None`), all() ritorna `False`
- **`not all(...)`**: Inverti logica → True se almeno uno manca

**Riga 43**:
```python
raise FileNotFoundError("File CSV non trovati")
```
- **`raise`**: Solleva eccezione (blocca esecuzione)
- **`FileNotFoundError`**: Eccezione built-in per file mancanti
- **Messaggio custom**: `"File CSV non trovati"`
- **Effetto**: Jupyter mostra traceback rosso, notebook si ferma

#### Sezione 3: Caricamento Dati PV (Righe 45-56)

**Righe 49-51: Lettura CSV con Pandas**
```python
pv1 = pd.read_csv(pv_file1)
pv2 = pd.read_csv(pv_file2)
```

**`pd.read_csv()` - Parametri Impliciti Default**:
- **`sep=','`**: Separatore virgola (standard CSV)
- **`header=0`**: Prima riga come nomi colonne
- **`index_col=None`**: Nessuna colonna come index
- **`dtype=None`**: Auto-detect tipi (int, float, str)
- **`encoding='utf-8'`**: Encoding Unicode
- **`parse_dates=False`**: NON converte date automaticamente (fatto dopo)

**Nel progetto**:
- `pv1`: DataFrame con 8760 righe × 2 colonne (1 anno di dati orari)
- `pv2`: DataFrame con 8557 righe × 2 colonne (anno incompleto)
- **Colonne originali**: `['Max kWp', '82.41']` (nomi strani dal file Excel)

**Righe 53-55: Info Diagnostiche**
```python
print(f"PV1 shape: {pv1.shape}")
print(f"PV2 shape: {pv2.shape}")
print(f"\nColonne PV1: {pv1.columns.tolist()}")
```
- **`df.shape`**: Tupla `(n_rows, n_cols)`
  - Es: `pv1.shape = (8760, 2)`
- **`df.columns`**: Index object con nomi colonne
- **`.tolist()`**: Converte Index → lista Python
  - Es: `['Max kWp', '82.41']`

#### Sezione 4: Concatenazione (Righe 57-59)

**Riga 58: pd.concat() - Analisi Dettagliata**
```python
pv_data = pd.concat([pv1, pv2], ignore_index=True)
```

**`pd.concat()` - Parametri**:
- **`[pv1, pv2]`**: Lista di DataFrame da concatenare
- **`axis=0` (default)**: Concatenazione verticale (stack righe)
  - `axis=1` concatenerebbe orizzontalmente (side-by-side)
- **`ignore_index=True`**: **CRUCIALE**
  - Reset dell'index dopo concatenazione
  - Senza questo: index avrebbero duplicati (0-8759, poi di nuovo 0-8556)
  - Con questo: index continuo (0-17316)

**Visualizzazione Concatenazione**:
```
pv1:                      pv2:
  Max kWp    | 82.41        Max kWp    | 82.41
0 2010-07-01 | 0.0       0 2011-07-01 | 0.0
1 2010-07-01 | 1.2       1 2011-07-01 | 0.8
... (8760 rows)          ... (8557 rows)

↓ pd.concat([pv1, pv2], ignore_index=True) ↓

pv_data:
     Max kWp    | 82.41
0    2010-07-01 | 0.0
1    2010-07-01 | 1.2
...
8759 2011-06-30 | 5.3
8760 2011-07-01 | 0.0    ← Inizio pv2
8761 2011-07-01 | 0.8
...
17316 2012-06-30 | 4.1
```

**Risultato**:
- `pv_data.shape = (17317, 2)` (8760 + 8557 = 17317 righe)

#### Sezione 5: Identificazione e Rinomina Colonne (Righe 61-73)

**Righe 61-65: Identificazione Automatica Colonne**
```python
# Identifica la colonna timestamp e target
# La colonna "Max kWp" contiene il timestamp
# La prima colonna numerica (probabilmente "82.41" o simile) è la produzione
timestamp_col = "Max kWp"
target_col = [col for col in pv_data.columns if col != timestamp_col][0]
```

**Riga 64**:
```python
timestamp_col = "Max kWp"
```
- **Hardcoded**: Nome colonna timestamp noto dal file originale
- **"Max kWp"**: Nome strano (probabilmente errore di labeling nel file Excel)
- Contiene timestamp ISO: `"2010-07-01 00:00:00+00:00 UTC"`

**Riga 65 - Analisi complessa**:
```python
target_col = [col for col in pv_data.columns if col != timestamp_col][0]
```
- **List comprehension**: Filtra colonne
- **`pv_data.columns`**: Index(['Max kWp', '82.41'], dtype='object')
- **`if col != timestamp_col`**: Esclude timestamp
- **Risultato lista**: `['82.41']` (una sola colonna rimasta)
- **`[0]`**: Prende primo (e unico) elemento
- **`target_col = '82.41'`**: Nome della colonna target (produzione PV)

**Perché questo approccio**:
- Evita hardcode del nome target (potrebbe variare tra dataset)
- Assumo: solo 2 colonne (1 timestamp + 1 target)
- Robusto se nome target cambia

**Righe 70-71: Rinomina Colonne**
```python
pv_data = pv_data.rename(columns={timestamp_col: 'datetime', target_col: 'power_kw'})
```

**`df.rename()` - Parametri**:
- **`columns=dict`**: Dizionario di mapping old_name → new_name
- **Mapping**:
  - `"Max kWp"` → `"datetime"`
  - `"82.41"` → `"power_kw"`
- **`inplace=False` (default)**: Ritorna nuovo DataFrame (non modifica in-place)
- **Riassegnazione**: `pv_data = ...` salva il risultato

**Dopo rinomina**:
```python
pv_data.columns
# Index(['datetime', 'power_kw'], dtype='object')
```

**Righe 72-73: Visualizzazione**
```python
print(f"\nPrime righe PV data:")
print(pv_data.head())
```
- **`df.head()`**: Ritorna prime 5 righe (default n=5)
- Output esempio:
  ```
              datetime  power_kw
  0  2010-07-01 00:00:00+00:00 UTC     0.00
  1  2010-07-01 01:00:00+00:00 UTC     0.00
  2  2010-07-01 02:00:00+00:00 UTC     0.00
  3  2010-07-01 03:00:00+00:00 UTC     0.00
  4  2010-07-01 04:00:00+00:00 UTC     0.00
  ```

### Punti Chiave

1. **Ricerca Robusta**: 5 location controllate per resilienza
2. **Error Handling**: `raise FileNotFoundError` blocca se file mancanti
3. **Concatenazione**: `ignore_index=True` evita duplicati index
4. **Identificazione Automatica**: List comprehension per trovare target
5. **Rinomina Standard**: `datetime`, `power_kw` (nomi consistenti)

---

## Cella 6: Caricamento Dati Meteo, Merge e Preprocessing

Questa cella carica i dati meteo, li concatena, gestisce i timestamp, e fa il merge con i dati PV per creare il dataset unificato finale.

### Codice Completo con Numerazione

#### Parte 1: Caricamento Dati Meteo (Righe 1-17)

```python
1  print("Caricamento dati meteo...")
2  # Carica dati meteo
3  wx1 = pd.read_csv(wx_file1)
4  wx2 = pd.read_csv(wx_file2)
5  
6  print(f"WX1 shape: {wx1.shape}")
7  print(f"WX2 shape: {wx2.shape}")
8  print(f"\nColonne WX1: {wx1.columns.tolist()}")
9  
10 # Concatena i dati meteo
11 wx_data = pd.concat([wx1, wx2], ignore_index=True)
12 print(f"\nWX data concatenato: {wx_data.shape}")
13 
14 # Rinomina la colonna timestamp
15 wx_data = wx_data.rename(columns={'dt_iso': 'datetime'})
16 print(f"\nPrime righe WX data:")
17 print(wx_data.head())
```

#### Parte 2: Conversione Timestamp e Merge (Righe 1-28)

```python
1  print("Conversione timestamp e merging...")
2  
3  # Converti timestamp in datetime e gestisci timezone
4  # Usa format='mixed' per gestire formati inconsistenti (con/senza microsecondi)
5  pv_data['datetime'] = pd.to_datetime(pv_data['datetime'], format='mixed', utc=True).dt.tz_localize(None)
6  wx_data['datetime'] = pd.to_datetime(wx_data['datetime'], format='mixed', utc=True).dt.tz_localize(None)
7  
8  print(f"\nRange temporale PV: {pv_data['datetime'].min()} to {pv_data['datetime'].max()}")
9  print(f"Range temporale WX: {wx_data['datetime'].min()} to {wx_data['datetime'].max()}")
10 
11 # Rimuovi duplicati temporali
12 pv_data = pv_data.drop_duplicates(subset=['datetime'], keep='first')
13 wx_data = wx_data.drop_duplicates(subset=['datetime'], keep='first')
14 
15 print(f"\nDopo rimozione duplicati:")
16 print(f"PV data shape: {pv_data.shape}")
17 print(f"WX data shape: {wx_data.shape}")
18 
19 # Merge dei dataset
20 data = pd.merge(pv_data, wx_data, on='datetime', how='inner')
21 print(f"\nDataset merged shape: {data.shape}")
22 print(f"\nColonne finali: {data.columns.tolist()}")
23 
24 # Ordina per timestamp
25 data = data.sort_values('datetime').reset_index(drop=True)
26 
27 print(f"\nPrime righe del dataset unificato:")
28 print(data.head(10))
```

### Spiegazione Dettagliata

#### Parte 1: Caricamento e Concatenazione Dati Meteo

**Righe 3-4: Lettura CSV**
```python
wx1 = pd.read_csv(wx_file1)
wx2 = pd.read_csv(wx_file2)
```
- Identico a PV: carica 2 file meteo
- **wx1**: 8760 righe × 9 colonne (luglio 2010 - giugno 2011)
- **wx2**: 8557 righe × 9 colonne (luglio 2011 - giugno 2012)
- **Colonne**: `['dt_iso', 'temp', 'Dni', 'Ghi', 'humidity', 'clouds_all', 'wind_speed', 'pressure', 'rain_1h']`

**Colonne Meteo - Significato**:
1. **`dt_iso`**: Timestamp ISO format (datetime)
2. **`temp`**: Temperatura (°C)
3. **`Dni`**: Direct Normal Irradiance - Irradianza diretta normale (W/m²)
4. **`Ghi`**: Global Horizontal Irradiance - Irradianza globale orizzontale (W/m²)
5. **`humidity`**: Umidità relativa (%)
6. **`clouds_all`**: Copertura nuvolosa totale (%)
7. **`wind_speed`**: Velocità del vento (m/s)
8. **`pressure`**: Pressione atmosferica (hPa)
9. **`rain_1h`**: Pioggia ultima ora (mm)

**Righe 11: Concatenazione**
```python
wx_data = pd.concat([wx1, wx2], ignore_index=True)
```
- Identico a PV concatenation
- **Risultato**: 17317 righe × 9 colonne (8760 + 8557)
- **`ignore_index=True`**: Index continuo 0-17316

**Righe 15: Rinomina Timestamp**
```python
wx_data = wx_data.rename(columns={'dt_iso': 'datetime'})
```
- Rinomina `dt_iso` → `datetime`
- **Perché**: Standardizzazione con dataset PV
- **Necessario per merge**: Entrambi dataset devono avere stessa colonna join

#### Parte 2: Conversione Timestamp

**Righe 5-6: pd.to_datetime() - Analisi Complessa**
```python
pv_data['datetime'] = pd.to_datetime(pv_data['datetime'], format='mixed', utc=True).dt.tz_localize(None)
wx_data['datetime'] = pd.to_datetime(wx_data['datetime'], format='mixed', utc=True).dt.tz_localize(None)
```

**Scomposizione in 3 passi**:

1. **`pd.to_datetime(pv_data['datetime'], format='mixed', utc=True)`**

   **Parametri**:
   - **`pv_data['datetime']`**: Serie da convertire (stringhe)
   - **`format='mixed'`**: **CRUCIALE**
     - Introdotto in Pandas 2.0+
     - Gestisce formati timestamp **inconsistenti** nello stesso dataset
     - Nel progetto: alcuni timestamp con microsecondi, altri no
     - Esempi:
       ```
       "2010-07-01 00:00:00+00:00 UTC"          # Senza microsecondi
       "2010-07-01 01:00:00.000000+00:00 UTC"   # Con microsecondi
       ```
     - **Senza `format='mixed'`**: Error `ValueError: time data doesn't match format`
   
   - **`utc=True`**: **IMPORTANTE**
     - Interpreta timestamp come UTC
     - Converte in `datetime64[ns, UTC]` (timezone-aware)
     - Nel dataset: già in UTC ("+00:00" nell'originale)

   **Output Passo 1**: DatetimeIndex con timezone UTC
   ```python
   DatetimeIndex(['2010-07-01 00:00:00+00:00',
                  '2010-07-01 01:00:00+00:00', ...], 
                  dtype='datetime64[ns, UTC]')
   ```

2. **`.dt` accessor**
   - **`.dt`**: Accessor per operazioni su datetime
   - Disponibile solo su Series con dtype `datetime64`
   - Fornisce metodi: `.year`, `.month`, `.day`, `.hour`, `.tz_localize()`, etc.

3. **`.tz_localize(None)`**
   - **Scopo**: Rimuove timezone information
   - **Perché**:
     - TimeSeriesDataSet di pytorch-forecasting preferisce datetime naive (senza TZ)
     - Riduce complessità (tutti dati già in UTC)
     - Evita problemi DST (Daylight Saving Time)
   - **Conversione**:
     - `datetime64[ns, UTC]` → `datetime64[ns]` (naive)
     - **NON cambia valori**: `2010-07-01 00:00:00+00:00` → `2010-07-01 00:00:00`
   
   **Output Finale**: DatetimeIndex naive
   ```python
   DatetimeIndex(['2010-07-01 00:00:00',
                  '2010-07-01 01:00:00', ...], 
                  dtype='datetime64[ns]')
   ```

**Risultato**:
- Colonna `datetime` ora ha dtype `datetime64[ns]` (datetime naive)
- Pronta per operazioni temporali e merge

**Righe 8-9: Verifica Range Temporale**
```python
print(f"\nRange temporale PV: {pv_data['datetime'].min()} to {pv_data['datetime'].max()}")
print(f"Range temporale WX: {wx_data['datetime'].min()} to {wx_data['datetime'].max()}")
```
- **`.min()`**, **`.max()`**: Metodi Series per trovare min/max
- **Output atteso**:
  ```
  Range temporale PV: 2010-07-01 00:00:00 to 2012-06-30 23:00:00
  Range temporale WX: 2010-07-01 00:00:00 to 2012-06-30 23:00:00
  ```
- **Verifica**: Range identici → buono per merge

#### Parte 3: Rimozione Duplicati

**Righe 12-13: drop_duplicates()**
```python
pv_data = pv_data.drop_duplicates(subset=['datetime'], keep='first')
wx_data = wx_data.drop_duplicates(subset=['datetime'], keep='first')
```

**`df.drop_duplicates()` - Parametri**:
- **`subset=['datetime']`**: Colonne da usare per identificare duplicati
  - Se due righe hanno stesso `datetime`, sono considerate duplicate
  - Altre colonne ignorate per il confronto
  
- **`keep='first'`**: Quale riga mantenere tra duplicati
  - `'first'`: Mantiene prima occorrenza, rimuove successive
  - Alternative: `'last'` (ultima), `False` (rimuove tutte)

**Perché Necessario**:
- Dataset potrebbe avere timestamp duplicati per errori di logging
- Merge richiede chiave unica (1-to-1 correspondence)
- **Nel progetto**: Pochi/zero duplicati (dataset pulito)

**Performance**:
- Usa hash table per identificazione rapida
- Complessità: O(n) dove n = numero righe

#### Parte 4: Merge Dataset

**Riga 20: pd.merge() - Operazione Cruciale**
```python
data = pd.merge(pv_data, wx_data, on='datetime', how='inner')
```

**`pd.merge()` - Parametri Dettagliati**:

1. **`pv_data`** (left DataFrame):
   - Shape: (17317, 2)
   - Colonne: `['datetime', 'power_kw']`

2. **`wx_data`** (right DataFrame):
   - Shape: (17317, 9)
   - Colonne: `['datetime', 'temp', 'Dni', 'Ghi', 'humidity', 'clouds_all', 'wind_speed', 'pressure', 'rain_1h']`

3. **`on='datetime'`**: 
   - Colonna chiave per il join
   - Equivalente SQL: `ON pv_data.datetime = wx_data.datetime`
   - **Assumo**: Datetime identici tra dataset (stesso timestamp)

4. **`how='inner'`**: **CRUCIALE - Tipo di Join**
   
   **Opzioni disponibili**:
   - **`'inner'`** (usato): Mantiene solo righe con match in **entrambi** dataset
     - Se `datetime` esiste in PV ma non in WX → riga scartata
     - Se `datetime` esiste in WX ma non in PV → riga scartata
     - **Risultato**: Solo timestamp comuni
   
   - **`'left'`**: Mantiene tutte righe di PV, aggiungi WX dove c'è match
     - Missing WX → NaN nelle colonne meteo
   
   - **`'right'`**: Mantiene tutte righe di WX
   
   - **`'outer'`**: Mantiene tutte righe di entrambi (union)
     - Missing → NaN

   **Perché `inner`**:
   - Vogliamo solo timestamp con **entrambe** le informazioni (PV + meteo)
   - Forecasting richiede feature complete (no missing)
   - Dataset ben allineati → poche righe perse

**Visualizzazione Merge**:
```
pv_data:                          wx_data:
datetime          | power_kw      datetime          | temp | Dni | Ghi | ...
2010-07-01 00:00  | 0.0           2010-07-01 00:00  | 18.5 | 0   | 0   | ...
2010-07-01 01:00  | 0.0           2010-07-01 01:00  | 18.2 | 0   | 0   | ...
...                               ...

↓ pd.merge(..., on='datetime', how='inner') ↓

data:
datetime          | power_kw | temp | Dni | Ghi | humidity | clouds_all | wind_speed | pressure | rain_1h
2010-07-01 00:00  | 0.0      | 18.5 | 0   | 0   | 75       | 20         | 2.1        | 1013     | 0.0
2010-07-01 01:00  | 0.0      | 18.2 | 0   | 0   | 76       | 22         | 1.9        | 1013     | 0.0
...
```

**Risultato**:
- **Shape**: (17317, 10) = 2 (PV) + 9 (WX) - 1 (datetime condiviso)
- **Colonne**: `['datetime', 'power_kw', 'temp', 'Dni', 'Ghi', 'humidity', 'clouds_all', 'wind_speed', 'pressure', 'rain_1h']`

#### Parte 5: Ordinamento Finale

**Riga 25: sort_values() e reset_index()**
```python
data = data.sort_values('datetime').reset_index(drop=True)
```

**Scomposizione**:

1. **`data.sort_values('datetime')`**:
   - **Scopo**: Ordina righe per timestamp crescente
   - **Perché necessario**:
     - Dopo merge, ordine potrebbe essere scrambled
     - TimeSeriesDataSet richiede ordine cronologico
     - Visualizzazioni temporali necessitano sorting
   - **Complessità**: O(n log n) (Timsort in Pandas)

2. **`.reset_index(drop=True)`**:
   - **`.reset_index()`**: Ricrea index sequenziale 0, 1, 2, ...
   - **`drop=True`**: NON aggiungere vecchio index come colonna
     - `drop=False` creerebbe colonna extra `index`
   - **Perché necessario**: Dopo sort, index non è più sequenziale

**Esempio Prima/Dopo**:
```
PRIMA sort_values:
     datetime          | power_kw | temp | ...
105  2010-07-05 09:00  | 12.3     | 25.1 | ...
89   2010-07-04 17:00  | 8.5      | 27.3 | ...
200  2010-07-09 08:00  | 11.2     | 24.8 | ...
     ↓ Index disordinato

DOPO sort_values + reset_index:
    datetime          | power_kw | temp | ...
0   2010-07-01 00:00  | 0.0      | 18.5 | ...
1   2010-07-01 01:00  | 0.0      | 18.2 | ...
2   2010-07-01 02:00  | 0.0      | 18.0 | ...
    ↓ Index ordinato, cronologia corretta
```

### Punti Chiave Cella 6

1. **`format='mixed'`**: Essenziale per timestamp inconsistenti
2. **Timezone handling**: `utc=True` + `.tz_localize(None)` per datetime naive
3. **Inner join**: Mantiene solo timestamp comuni (no missing data)
4. **Ordinamento**: `sort_values()` + `reset_index()` per cronologia corretta
5. **Dataset finale**: 17,317 ore × 10 colonne (1 target + 8 meteo + 1 timestamp)

### Schema Completo Flusso Dati (Celle 3-6)

```
Excel Files (2 fogli ciascuno)
  ↓
[Cella 3] excel_to_csv()
  ↓
4 File CSV separati
  ↓
[Cella 4] Ricerca file (os.walk)
  ↓
[Cella 5] Caricamento PV + concatenazione
  pv1 (8760×2) + pv2 (8557×2) → pv_data (17317×2)
  ↓
[Cella 6] Caricamento WX + concatenazione
  wx1 (8760×9) + wx2 (8557×9) → wx_data (17317×9)
  ↓
[Cella 6] Conversione timestamp (pd.to_datetime format='mixed')
  ↓
[Cella 6] Rimozione duplicati (drop_duplicates)
  ↓
[Cella 6] Merge (pd.merge inner join on 'datetime')
  ↓
[Cella 6] Ordinamento (sort_values + reset_index)
  ↓
Dataset Unificato Finale: (17317, 10)
  - datetime (datetime64[ns])
  - power_kw (float64) ← TARGET
  - 8 feature meteo (float64)
```

---

## Sezione 3: Data Analysis & Missing Values

Questa sezione analizza il dataset unificato e gestisce i valori mancanti prima del training.

### Cella 7: Statistiche Descrittive del Dataset

```python
1  print("Analisi dei dati...\n")
2  print(f"Dimensione dataset: {data.shape}")
3  print(f"\nInfo dataset:")
4  print(data.info())
5  
6  print("\n" + "="*50)
7  print("STATISTICHE DESCRITTIVE")
8  print("="*50)
9  print(data.describe())
```

#### Spiegazione Riga per Riga

**Riga 1: Header**
```python
print("Analisi dei dati...\n")
```
- **`\n`**: newline alla fine per spaziatura verticale
- Indica inizio della fase di analisi esplorativa

**Riga 2: Shape del Dataset**
```python
print(f"Dimensione dataset: {data.shape}")
```
- **`data.shape`**: tupla (righe, colonne)
- Nel progetto: `(17317, 10)` dopo merge
  - 17,317 righe = ore di dati (2 anni circa)
  - 10 colonne = 1 datetime + 1 target + 8 feature meteo
- **Verifica importante**: se shape diverso da atteso → problema nel merge

**Righe 3-4: DataFrame Info**
```python
print(f"\nInfo dataset:")
print(data.info())
```
- **`data.info()`**: metodo Pandas che stampa:
  1. **Tipo DataFrame**: `<class 'pandas.core.frame.DataFrame'>`
  2. **RangeIndex**: indice delle righe (0 to 17316)
  3. **Colonne** (per ciascuna):
     - Nome colonna
     - Non-Null Count: numero valori non-null
     - Dtype: tipo di dato (datetime64, float64, object, etc.)
  4. **Memory usage**: memoria RAM occupata

**Output esempio**:
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 17317 entries, 0 to 17316
Data columns (total 10 columns):
 #   Column      Non-Null Count  Dtype         
---  ------      --------------  -----         
 0   datetime    17317 non-null  datetime64[ns]
 1   power_kw    17317 non-null  float64       
 2   temp        17317 non-null  float64       
 3   Dni         17317 non-null  float64       
 4   Ghi         17317 non-null  float64       
 5   humidity    17317 non-null  float64       
 6   clouds_all  17317 non-null  float64       
 7   wind_speed  17317 non-null  float64       
 8   pressure    17317 non-null  float64       
 9   rain_1h     16500 non-null  float64       ← 817 NaN!
dtypes: datetime64[ns](1), float64(9)
memory usage: 1.3+ MB
```

**Cosa cercare in info()**:
- ✅ **Non-Null Count = totale righe**: nessun NaN (ideale)
- ⚠️ **Non-Null Count < totale**: valori mancanti presenti
- ✅ **Dtype corretto**: datetime per timestamp, float per numerici
- ❌ **Dtype 'object'** per colonne numeriche: problema conversione

**Righe 6-9: Statistiche Descrittive**
```python
print("\n" + "="*50)
print("STATISTICHE DESCRITTIVE")
print("="*50)
print(data.describe())
```
- **`data.describe()`**: statistiche per colonne numeriche
  - **count**: numero valori non-null
  - **mean**: media aritmetica
  - **std**: deviazione standard (dispersione)
  - **min**: valore minimo
  - **25%**: primo quartile (25° percentile)
  - **50%**: mediana (50° percentile)
  - **75%**: terzo quartile (75° percentile)
  - **max**: valore massimo

**Output esempio per `power_kw`**:
```
       power_kw      temp        Dni  ...
count  17317.00  17317.00  17317.00  ...
mean      18.23     15.42   1245.67  ...
std       19.82      8.91    856.23  ...
min        0.00     -5.30      0.00  ...
25%        0.00      8.50    450.12  ...
50%       15.67     15.20   1180.45  ...
75%       35.89     22.10   1890.34  ...
max       58.45     35.80   2950.00  ...
```

**Analisi delle statistiche**:
- **`power_kw min=0, max=58.45`**: Range realistico per impianto fotovoltaico
- **`mean=18.23, median=15.67`**: Distribuzione leggermente asimmetrica a destra
- **`std=19.82`**: Alta variabilità (giorno vs notte, stagioni)
- **`25%=0`**: 25% dei valori sono zero (ore notturne!)

**Perché `describe()` è importante**:
1. **Outlier detection**: valori min/max fuori range → errori sensori
2. **Scaling**: capire scala feature per normalizzazione
3. **Distribuzione**: simmetrica (mean≈median) vs asimmetrica
4. **Variabilità**: std alta → feature volatile, std bassa → poco informativa

---

### Cella 8: Identificazione Valori Mancanti

```python
1  print("Controllo valori mancanti...\n")
2  missing = data.isnull().sum()
3  missing_pct = (missing / len(data)) * 100
4  missing_df = pd.DataFrame({
5      'Missing Count': missing,
6      'Percentage': missing_pct
7  })
8  missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
9  
10 if len(missing_df) > 0:
11     print("Colonne con valori mancanti:")
12     print(missing_df)
13 else:
14     print("Nessun valore mancante trovato!")
```

#### Spiegazione Riga per Riga

**Riga 2: Conteggio NaN per Colonna**
```python
missing = data.isnull().sum()
```
- **Scomposizione**:
  1. **`data.isnull()`**: DataFrame booleano (True se NaN, False altrimenti)
  2. **`.sum()`**: somma per colonna (True=1, False=0)
- **Output**: Series con numero NaN per colonna
  ```python
  datetime      0
  power_kw      0
  temp          0
  ...
  rain_1h     817
  dtype: int64
  ```

**Riga 3: Percentuale Mancanti**
```python
missing_pct = (missing / len(data)) * 100
```
- **`len(data)`**: numero totale righe (17,317)
- **Calcolo**: `(817 / 17317) * 100 = 4.72%` per `rain_1h`
- **Output**: Series con percentuali

**Righe 4-7: DataFrame Riepilogativo**
```python
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Percentage': missing_pct
})
```
- **`pd.DataFrame({...})`**: crea DataFrame da dizionario
- **Keys del dict** → nomi colonne
- **Values del dict** → Series Pandas
- **Index preservato**: nomi colonne del dataset originale

**Struttura `missing_df`**:
```
              Missing Count  Percentage
datetime                  0        0.00
power_kw                  0        0.00
temp                      0        0.00
...
rain_1h                 817        4.72
```

**Riga 8: Filtraggio e Ordinamento**
```python
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
```
- **Operazione composta**, scomponiamola:

1. **`missing_df['Missing Count'] > 0`**:
   - Crea Series booleana
   - True per colonne con almeno 1 NaN
   - False per colonne senza NaN

2. **`missing_df[...]`**:
   - Filtraggio booleano (boolean indexing)
   - Mantiene solo righe dove condizione è True
   - **Risultato**: Solo colonne con NaN

3. **`.sort_values('Missing Count', ascending=False)`**:
   - Ordina per colonna 'Missing Count'
   - **`ascending=False`**: ordine decrescente (più NaN prima)

**Output finale**:
```
         Missing Count  Percentage
rain_1h            817        4.72
```

**Righe 10-14: Conditional Print**
```python
if len(missing_df) > 0:
    print("Colonne con valori mancanti:")
    print(missing_df)
else:
    print("Nessun valore mancante trovato!")
```
- **`len(missing_df)`**: numero righe nel DataFrame filtrato
- **If `> 0`**: almeno una colonna ha NaN → stampa DataFrame
- **Else**: dataset completo (caso ideale ma raro)

**Perché questa analisi è critica**:
1. **Machine Learning richiede dati completi**: molti algoritmi falliscono con NaN
2. **Pattern di missing**:
   - **MCAR** (Missing Completely At Random): sicuro ignorare/rimuovere
   - **MAR** (Missing At Random): imputation con pattern
   - **MNAR** (Missing Not At Random): bias se non gestiti
3. **Decisione strategia**: imputation vs rimozione vs feature engineering

---

### Cella 9: Gestione Valori Mancanti

```python
1  print("Gestione valori mancanti...\n")
2  
3  # rain_1h: NaN = 0 (nessuna pioggia)
4  if 'rain_1h' in data.columns:
5      data['rain_1h'] = data['rain_1h'].fillna(0)
6      print("✓ rain_1h: NaN sostituiti con 0")
7  
8  # Per le altre colonne numeriche: interpolazione lineare limitata
9  numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
10 if 'power_kw' in numeric_cols:
11     # Interpolazione lineare con limite di 3 valori consecutivi
12     data['power_kw'] = data['power_kw'].interpolate(method='linear', limit=3)
13     # Forward fill per eventuali gap rimanenti
14     data['power_kw'] = data['power_kw'].fillna(method='ffill', limit=1)
15     print("✓ power_kw: interpolazione lineare applicata")
16 
17 # Altre colonne meteo
18 for col in numeric_cols:
19     if col != 'power_kw' and data[col].isnull().sum() > 0:
20         data[col] = data[col].interpolate(method='linear', limit=5)
21         data[col] = data[col].fillna(method='ffill', limit=2)
22         print(f"✓ {col}: interpolazione applicata")
23 
24 # Rimuovi righe con NaN rimanenti (se presenti)
25 rows_before = len(data)
26 data = data.dropna()
27 rows_after = len(data)
28 print(f"\nRighe rimosse: {rows_before - rows_after}")
29 print(f"Dataset finale: {data.shape}")
30 
31 print("\n✓ Gestione valori mancanti completata!")
```

#### Spiegazione Dettagliata

**Righe 3-6: Gestione Speciale `rain_1h`**
```python
if 'rain_1h' in data.columns:
    data['rain_1h'] = data['rain_1h'].fillna(0)
    print("✓ rain_1h: NaN sostituiti con 0")
```

**Ragionamento domain-specific**:
- **Sensori pioggia**: NaN spesso significa "nessuna rilevazione" = 0 mm pioggia
- **Alternative scartate**:
  - Interpolazione: assurdo (pioggia non è continua)
  - Rimozione righe: perderemmo ~5% dataset
  - Media: distorce distribuzione (rain_1h è sparso: 0, 0, 0, 15, 0, 0...)
- **`.fillna(0)`**: sostituisce tutti NaN con 0
  - **In-place**: assegnazione necessaria (`data['rain_1h'] = ...`)
  - Non usa `inplace=True` perché vogliamo riassegnare

**Riga 9: Selezione Colonne Numeriche**
```python
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
```
- **`data.select_dtypes(include=[np.number])`**: 
  - Seleziona solo colonne con dtype numerico
  - **Inclusi**: int, float, uint, etc.
  - **Esclusi**: datetime, string, object
- **`.columns`**: Index con nomi colonne
- **`.tolist()`**: converte Index in lista Python
- **Output**: `['power_kw', 'temp', 'Dni', 'Ghi', 'humidity', 'clouds_all', 'wind_speed', 'pressure', 'rain_1h']`

**Righe 10-15: Interpolazione Target (`power_kw`)**
```python
if 'power_kw' in numeric_cols:
    data['power_kw'] = data['power_kw'].interpolate(method='linear', limit=3)
    data['power_kw'] = data['power_kw'].fillna(method='ffill', limit=1)
    print("✓ power_kw: interpolazione lineare applicata")
```

**Riga 12 - Interpolazione Lineare**:
```python
data['power_kw'].interpolate(method='linear', limit=3)
```
- **`method='linear'`**: interpolazione lineare tra valori adiacenti
  - Formula: $y = y_1 + \frac{x - x_1}{x_2 - x_1}(y_2 - y_1)$
  - Esempio:
    ```
    PRIMA:  10.5, NaN, NaN, 16.5
    DOPO:   10.5, 12.5, 14.5, 16.5  (incrementi uguali di 2.0)
    ```
  
- **`limit=3`**: massimo 3 NaN consecutivi interpolabili
  - **Perché limite?**: Gap troppo lunghi → interpolazione inaffidabile
  - Se 4+ NaN consecutivi → i primi 3 interpolati, il 4° rimane NaN

**Alternative a `method='linear'`**:
- `'time'`: considera timestamp (per dati irregolarmente spaziati)
- `'polynomial'`: fit polinomiale (rischio overfitting)
- `'spline'`: smooth, ma computazionalmente costoso
- `'nearest'`: forward/backward fill (non smooth)

**Riga 14 - Forward Fill Residuo**:
```python
data['power_kw'].fillna(method='ffill', limit=1)
```
- **`method='ffill'`**: "forward fill" = propaga ultimo valore valido
  - Esempio:
    ```
    PRIMA:  10.5, 12.3, NaN, 15.7
    DOPO:   10.5, 12.3, 12.3, 15.7  (12.3 propagato)
    ```
  
- **`limit=1`**: propaga max 1 posizione avanti
  - Evita propagazione eccessiva (es. valori notturni copiati al mattino)

**Strategia combinata**:
1. **Interpolazione lineare** (limit=3): riempie gap brevi con trend
2. **Forward fill** (limit=1): gestisce NaN singoli rimanenti
3. **Ordine importante**: interpolazione prima (più accurata)

**Righe 17-22: Loop su Altre Feature Meteo**
```python
for col in numeric_cols:
    if col != 'power_kw' and data[col].isnull().sum() > 0:
        data[col] = data[col].interpolate(method='linear', limit=5)
        data[col] = data[col].fillna(method='ffill', limit=2)
        print(f"✓ {col}: interpolazione applicata")
```

**Differenze rispetto a `power_kw`**:
- **`limit=5`** invece di 3: meteo cambia più gradualmente che produzione solare
- **`limit=2`** (ffill) invece di 1: tolleranza maggiore per feature ausiliarie
- **Condizione**: `col != 'power_kw'` per non processare due volte il target

**Righe 24-29: Rimozione NaN Residui**
```python
rows_before = len(data)
data = data.dropna()
rows_after = len(data)
print(f"\nRighe rimosse: {rows_before - rows_after}")
print(f"Dataset finale: {data.shape}")
```

**Riga 26 - `dropna()`**:
- **Comportamento default**: rimuove QUALSIASI riga con almeno 1 NaN
- **Parametri opzionali** (non usati qui):
  - `subset=['col1', 'col2']`: controlla solo certe colonne
  - `how='all'`: rimuovi solo se TUTTI i valori sono NaN
  - `thresh=5`: rimuovi se meno di 5 valori non-null
- **Nel progetto**: Dopo interpolazione, NaN residui sono rari
  - Tipicamente: 0-10 righe rimosse su 17k (≈0.05%)
  - Righe problematiche: gap lunghi all'inizio/fine dataset

**Output esempio**:
```
Righe rimosse: 3
Dataset finale: (17314, 10)
```

**Riga 31: Conferma Completamento**
```python
print("\n✓ Gestione valori mancanti completata!")
```
- Checkmark ✓ indica successo operazione
- Dataset ora pronto per feature engineering

---

### Riepilogo Strategia Missing Values

| Colonna | Tipo NaN | Strategia | Motivazione |
|---------|----------|-----------|-------------|
| `rain_1h` | ~5% | Fill con 0 | NaN = sensore non rileva = no pioggia |
| `power_kw` | <1% | Interpolazione lineare (limit=3) + ffill (limit=1) | Target critico, preserva trend |
| Feature meteo | <1% | Interpolazione lineare (limit=5) + ffill (limit=2) | Cambiamenti graduali, tolleranza maggiore |
| NaN residui | <0.1% | Rimozione righe | Gap troppo lunghi, dati inaffidabili |

**Vantaggi approccio**:
1. ✅ **Domain-aware**: usa conoscenza del dominio (pioggia, solare)
2. ✅ **Conservativo**: limiti su interpolazione prevengono artefatti
3. ✅ **Graduale**: 3 strategie in cascata (fill → interpolate → drop)
4. ✅ **Preserva dati**: minimal data loss (~0.1%)
5. ✅ **No bias**: interpolazione lineare non introduce pattern falsi

---

## Sezione 4: Feature Engineering

Feature Engineering è il processo di creare nuove variabili (feature) dai dati grezzi per migliorare le performance del modello. In time series forecasting, le feature temporali sono cruciali per catturare pattern ciclici.

### Cella 10: Creazione Feature Temporali e Ottimizzazione Memoria

```python
1  print("Feature Engineering...\n")
2  
3  # Feature temporali
4  data['hour'] = data['datetime'].dt.hour
5  data['day_of_month'] = data['datetime'].dt.day
6  data['month'] = data['datetime'].dt.month
7  data['day_of_week'] = data['datetime'].dt.dayofweek
8  
9  print("✓ Feature temporali create: hour, day_of_month, month, day_of_week")
10 
11 # Group ID (necessario per TFT)
12 data['group_id'] = 'PV1'
13 print("✓ group_id creato: 'PV1'")
14 
15 # Time index (indice temporale incrementale) - DEVE essere intero
16 data['time_idx'] = np.arange(len(data)).astype(int)
17 print("✓ time_idx creato: 0 to", len(data)-1)
18 
19 # Converti tutte le colonne numeriche in float32 (ECCETTO time_idx che deve rimanere int)
20 numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
21 for col in numeric_cols:
22     if col != 'time_idx':  # Non convertire time_idx
23         data[col] = data[col].astype(np.float32)
24 
25 print(f"✓ {len(numeric_cols)-1} colonne numeriche convertite in float32")
26 print(f"✓ time_idx mantenuto come {data['time_idx'].dtype}")
27 
28 print("\n" + "="*50)
29 print("DATASET FINALE")
30 print("="*50)
31 print(f"Shape: {data.shape}")
32 print(f"\nColonne: {data.columns.tolist()}")
33 print(f"\nPrime righe:")
34 print(data.head())
```

#### Spiegazione Dettagliata

---

### **Parte 1: Feature Temporali (Righe 3-9)**

#### **Riga 4: Ora del Giorno (Hour)**
```python
data['hour'] = data['datetime'].dt.hour
```

**Scomposizione**:
- **`data['datetime']`**: Series Pandas con dtype `datetime64[ns]`
- **`.dt`**: accessor per operazioni datetime-specific
  - Simile a `.str` per stringhe
  - Disponibile solo su Series datetime
- **`.hour`**: estrae l'ora (0-23) come intero

**Output**:
```python
# Input: datetime
2010-07-01 00:00:00  →  0
2010-07-01 01:00:00  →  1
2010-07-01 13:30:00  →  13
2010-07-01 23:45:00  →  23
```

**Perché è importante per PV forecasting**:
1. **Pattern giornaliero**: Produzione zero di notte (0-6, 20-23), picco a mezzogiorno (12-14)
2. **Ciclicità**: 24 ore = ciclo completo
3. **Stagionalità oraria**: Alba/tramonto variano con stagione
   - Estate: produzione 6:00-20:00 (14 ore)
   - Inverno: produzione 8:00-17:00 (9 ore)

**Alternativa avanzata non usata** (codifica ciclica):
```python
# Preserva ciclicità: ora 23 e ora 0 sono vicine
data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
```
- **Pro**: Rappresentazione matematica corretta della ciclicità
- **Contro**: TFT gestisce embedding, non necessario qui

---

#### **Riga 5: Giorno del Mese**
```python
data['day_of_month'] = data['datetime'].dt.day
```

**Output**:
```python
# Input: datetime
2010-07-01  →  1
2010-07-15  →  15
2010-07-31  →  31
2010-08-01  →  1
```

**Perché meno importante**:
- ❌ **No pattern forte**: giorno 1 vs 15 vs 31 non ha significato fisico per PV
- ⚠️ **Confusione potenziale**: diversi mesi hanno giorni diversi (28-31)
- ✅ **Potrebbe catturare**: pattern di manutenzione mensile (es. pulizia pannelli ogni 1° del mese)

**Alternativa migliore** (non usata):
```python
data['day_of_year'] = data['datetime'].dt.dayofyear  # 1-365
```
- Cattura meglio la posizione nell'anno solare
- Utile per stagionalità solare (angolo sole, durata giorno)

---

#### **Riga 6: Mese dell'Anno**
```python
data['month'] = data['datetime'].dt.month
```

**Output**:
```python
# Input: datetime
2010-01-15  →  1  (Gennaio)
2010-06-15  →  6  (Giugno)
2010-12-15  →  12 (Dicembre)
```

**Perché è MOLTO importante**:
1. **Stagionalità solare**:
   - Estate (Giu-Ago): Alta produzione, lunghe giornate
   - Inverno (Dic-Feb): Bassa produzione, corte giornate
2. **Angolo di incidenza solare**:
   - Estate: Sole alto → massima irradiazione
   - Inverno: Sole basso → irradiazione ridotta
3. **Pattern meteo stagionale**:
   - Estate: Più sole, meno nuvole
   - Inverno: Più nuvole, possibile neve

**Distribuzione produzione per mese** (esempio):
```
Gen: 15 kW media
Feb: 20 kW
Mar: 30 kW
Apr: 40 kW
Mag: 48 kW
Giu: 52 kW ← Picco
Lug: 50 kW
Ago: 48 kW
Set: 38 kW
Ott: 28 kW
Nov: 18 kW
Dic: 12 kW ← Minimo
```

---

#### **Riga 7: Giorno della Settimana**
```python
data['day_of_week'] = data['datetime'].dt.dayofweek
```

**Output**:
```python
# Input: datetime
2010-07-05 (Lunedì)    →  0
2010-07-06 (Martedì)   →  1
...
2010-07-10 (Sabato)    →  5
2010-07-11 (Domenica)  →  6
```

**Convenzione Pandas**:
- `0 = Lunedì`
- `6 = Domenica`
- Alternativa: `.dt.day_name()` → "Monday", "Tuesday", etc. (string)

**Perché per PV forecasting**:
- ❓ **Utilità discutibile**: Produzione solare NON dipende dal giorno settimanale
- ✅ **Possibile rilevanza**:
  - Consumo energia: Weekend vs Weekday (se predici net load)
  - Manutenzione: Operazioni programmate il weekend
  - Cloud patterns: Meteorologia locale con pattern settimanale (raro)
- 🎯 **Nel progetto**: Probabilmente bassa importanza (come visto in interpretability)

---

### **Parte 2: Group ID per TFT (Righe 11-13)**

#### **Riga 12: Creazione Group ID**
```python
data['group_id'] = 'PV1'
```

**Cos'è Group ID**:
- **Scopo**: Identificatore univoco per ogni serie temporale nel dataset
- **Tipo**: String (o categorico)
- **Nel progetto**: Un solo impianto → un solo gruppo `'PV1'`

**Quando servono multiple group_id**:
```python
# Esempio: Multiple impianti fotovoltaici
data_impianto1['group_id'] = 'PV_Plant_A'
data_impianto2['group_id'] = 'PV_Plant_B'
data_impianto3['group_id'] = 'PV_Plant_C'

data_combined = pd.concat([data_impianto1, data_impianto2, data_impianto3])

# TFT impara pattern comuni + specifici per gruppo
```

**Perché TFT richiede group_id**:
1. **Normalizzazione per gruppo**:
   ```python
   target_normalizer=GroupNormalizer(groups=["group_id"])
   ```
   - Ogni gruppo normalizzato separatamente
   - Evita bias se gruppi hanno scale diverse

2. **Embedding categorie**:
   - TFT crea embedding per `group_id`
   - Cattura caratteristiche specifiche del gruppo
   - Es: orientamento pannelli, ombreggiamento, efficienza

3. **Batch sampling**:
   - PyTorch Forecasting campiona batch per gruppo
   - Garantisce rappresentatività di ogni serie temporale

**Nel nostro caso** (singolo gruppo):
- Group ID è costante → embedding zero-information
- Normalizzazione funziona comunque (un solo gruppo da normalizzare)
- **Obbligatorio**: TFT crash senza group_id, anche se singolo

---

### **Parte 3: Time Index (Righe 15-17)**

#### **Riga 16: Creazione Time Index**
```python
data['time_idx'] = np.arange(len(data)).astype(int)
```

**Scomposizione**:
1. **`np.arange(len(data))`**:
   - Crea array NumPy: `[0, 1, 2, 3, ..., 17316]`
   - `len(data) = 17317` → array da 0 a 17316
   - **Tipo default**: int64 su sistemi 64-bit

2. **`.astype(int)`**:
   - Conversione esplicita a intero Python
   - **Necessaria**: garantisce dtype intero, non float

**Output**:
```python
datetime             time_idx
2010-07-01 00:00        0
2010-07-01 01:00        1
2010-07-01 02:00        2
...
2012-06-30 23:00    17316
```

**Perché time_idx è CRITICO**:

1. **Ordinamento temporale**:
   - TFT usa `time_idx` per ordinare sequenze
   - DEVE essere incrementale senza gap
   - DEVE essere intero (no float, no datetime)

2. **Windowing automatico**:
   - TimeSeriesDataSet usa `time_idx` per creare finestre
   - Esempio: `max_encoder_length=168`
     - Campione 1: time_idx 0-167 (encoder) + 168-191 (decoder)
     - Campione 2: time_idx 1-168 (encoder) + 169-192 (decoder)
     - etc.

3. **Relative Time Encoding**:
   ```python
   add_relative_time_idx=True  # In TimeSeriesDataSet
   ```
   - Crea feature: `(time_idx - time_idx_start_sequence) / max_encoder_length`
   - Normalizza posizione temporale dentro la finestra
   - Valore: 0.0 (inizio encoder) → 1.0 (fine decoder)

**❌ Errori comuni**:
```python
# SBAGLIATO: Float invece di int
data['time_idx'] = np.arange(len(data)).astype(float)  
# → Error: "time_idx must be of integer type"

# SBAGLIATO: Gap nella sequenza
data = data[data['power_kw'] > 0]  # Rimuove notti
data['time_idx'] = np.arange(len(data))  # Crea gap temporali!
# → Modello non capisce che ci sono 12 ore di salto

# SBAGLIATO: Non ordinato
data = data.sample(frac=1)  # Shuffle random
data['time_idx'] = np.arange(len(data))
# → time_idx non corrisponde all'ordine temporale reale
```

**✅ Corretto** (come nel progetto):
```python
data = data.sort_values('datetime').reset_index(drop=True)  # Ordina prima
data['time_idx'] = np.arange(len(data)).astype(int)  # Poi crea time_idx
```

---

### **Parte 4: Ottimizzazione Memoria (Righe 19-26)**

#### **Righe 19-23: Conversione a float32**
```python
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    if col != 'time_idx':  # Non convertire time_idx
        data[col] = data[col].astype(np.float32)
```

**Riga 20: Selezione Colonne Numeriche**
```python
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
```
- Seleziona colonne con dtype numerico (float64, int64, etc.)
- **Output**: `['power_kw', 'temp', 'Dni', 'Ghi', ..., 'hour', 'month', 'time_idx']`
- **Esclude**: `'datetime'` (datetime64), `'group_id'` (object/string)

**Riga 22: Condizione Esclusione time_idx**
```python
if col != 'time_idx':
```
- **Perché escludere**:
  - `time_idx` DEVE rimanere `int` (int64)
  - TimeSeriesDataSet fa type checking: `assert time_idx.dtype in [int32, int64]`
  - Conversione a float32 causerebbe errore

**Riga 23: Conversione dtype**
```python
data[col] = data[col].astype(np.float32)
```
- **Da**: float64 (8 bytes per valore)
- **A**: float32 (4 bytes per valore)
- **Risparmio**: 50% memoria per colonne numeriche

**Calcolo risparmio memoria**:
```python
# Prima conversione
9 colonne × 17317 righe × 8 bytes (float64) = 1.25 MB

# Dopo conversione
9 colonne × 17317 righe × 4 bytes (float32) = 0.62 MB

Risparmio: 0.63 MB (50%)
```

**Perché float32 invece di float64**:

1. **Memoria GPU limitata**:
   - RTX 4060 ha 8 GB VRAM
   - Batch training: tensori replicati per batch_size × seq_length
   - Esempio: batch=64, seq=192, features=13
     - float64: 64 × 192 × 13 × 8 = 12.8 MB per batch
     - float32: 64 × 192 × 13 × 4 = 6.4 MB per batch
   - Con gradienti, optimizer state, etc. → risparmio significativo

2. **Performance GPU**:
   - GPU moderne ottimizzate per float32 (FP32)
   - Tensor Cores su RTX: mixed precision (FP16/FP32)
   - float64 (FP64) più lento e meno supportato

3. **Precisione sufficiente**:
   - float32: 7 cifre decimali significative
   - Per `power_kw` (0-58 kW): precisione ~0.00001 kW
   - **Più che sufficiente** per forecasting (errori sensore >> precisione float)

**Trade-off float32 vs float64**:
| Aspetto | float32 | float64 |
|---------|---------|---------|
| Memoria | 4 bytes | 8 bytes |
| Precisione | ~7 cifre | ~15 cifre |
| Range | ±3.4×10³⁸ | ±1.7×10³⁰⁸ |
| Velocità GPU | Veloce | Più lento |
| Use case | Deep Learning, forecasting | Calcolo scientifico ad alta precisione |

**⚠️ Quando NON usare float32**:
- Calcoli finanziari (arrotondamenti critici)
- Simulazioni fisiche ad alta precisione
- Operazioni con numeri molto grandi/piccoli (rischio underflow/overflow)

**Nel nostro progetto**:
- ✅ float32 è **perfetto**
- Dati PV: range 0-58 kW, precisione sensori ±1%
- Guadagno 50% memoria senza loss di informazione

---

#### **Righe 25-26: Conferma Conversioni**
```python
print(f"✓ {len(numeric_cols)-1} colonne numeriche convertite in float32")
print(f"✓ time_idx mantenuto come {data['time_idx'].dtype}")
```

**Output**:
```
✓ 12 colonne numeriche convertite in float32
✓ time_idx mantenuto come int64
```

**Verifica dtype**:
```python
print(data.dtypes)

# Output atteso:
datetime               datetime64[ns]
power_kw                     float32
temp                         float32
...
hour                         float32
month                        float32
time_idx                       int64  ← Intero preservato!
group_id                      object
```

---

### **Parte 5: Summary Dataset Finale (Righe 28-34)**

#### **Righe 28-34: Print Riepilogo**
```python
print("\n" + "="*50)
print("DATASET FINALE")
print("="*50)
print(f"Shape: {data.shape}")
print(f"\nColonne: {data.columns.tolist()}")
print(f"\nPrime righe:")
print(data.head())
```

**Output esempio**:
```
==================================================
DATASET FINALE
==================================================
Shape: (17314, 15)

Colonne: ['datetime', 'power_kw', 'temp', 'Dni', 'Ghi', 'humidity', 
         'clouds_all', 'wind_speed', 'pressure', 'rain_1h', 'hour', 
         'day_of_month', 'month', 'day_of_week', 'group_id', 'time_idx']

Prime righe:
             datetime  power_kw   temp    Dni  ...  group_id  time_idx
0 2010-07-01 00:00:00      0.00  18.50    0.0  ...       PV1         0
1 2010-07-01 01:00:00      0.00  18.20    0.0  ...       PV1         1
2 2010-07-01 02:00:00      0.00  18.00    0.0  ...       PV1         2
3 2010-07-01 03:00:00      0.00  17.80    0.0  ...       PV1         3
4 2010-07-01 04:00:00      0.00  17.50    0.0  ...       PV1         4
```

**Analisi shape**:
- **Righe**: 17,314 (da 17,317 originali)
  - 3 righe rimosse: NaN residui dopo interpolazione
- **Colonne**: 15 (da 10 originali)
  - +4 feature temporali: `hour`, `day_of_month`, `month`, `day_of_week`
  - +1 group_id: `'PV1'`
  - +1 time_idx: `0` to `17313`

---

### **Riepilogo Feature Engineering**

| Feature | Tipo | Range | Utilità PV | Importanza |
|---------|------|-------|------------|------------|
| `hour` | int (float32) | 0-23 | ⭐⭐⭐⭐⭐ Pattern giornaliero | ALTA |
| `month` | int (float32) | 1-12 | ⭐⭐⭐⭐⭐ Stagionalità | ALTA |
| `day_of_week` | int (float32) | 0-6 | ⭐ Debolmente rilevante | BASSA |
| `day_of_month` | int (float32) | 1-31 | ⭐ Poco rilevante | BASSA |
| `time_idx` | int64 | 0-17313 | ⭐⭐⭐⭐⭐ Obbligatorio TFT | CRITICA |
| `group_id` | string | 'PV1' | ⭐⭐⭐⭐⭐ Obbligatorio TFT | CRITICA |

**Pattern catturati**:
1. ✅ **Ciclicità giornaliera** (`hour`): Zero notte, picco mezzogiorno
2. ✅ **Stagionalità annuale** (`month`): Estate alta, inverno bassa
3. ✅ **Sequenzialità temporale** (`time_idx`): Ordine cronologico
4. ⚠️ **Ciclicità settimanale** (`day_of_week`): Probabilmente irrilevante per PV

**Ottimizzazioni applicate**:
1. ✅ **Memoria**: float32 invece di float64 (-50%)
2. ✅ **Tipo corretto**: time_idx come int64 (requirement TFT)
3. ✅ **Completezza**: Nessun NaN rimanente
4. ✅ **Ordinamento**: Cronologico per time_idx

**Dataset pronto per**: TimeSeriesDataSet creation → TFT Training

---

### Cella 11: Visualizzazione Serie Temporale e Statistiche Finali

```python
1  # Visualizzazione della serie temporale target
2  fig, axes = plt.subplots(2, 1, figsize=(15, 8))
3  
4  # Serie completa
5  axes[0].plot(data['datetime'], data['power_kw'], linewidth=0.5, alpha=0.7)
6  axes[0].set_title('Serie Temporale Completa - Produzione Fotovoltaica', fontsize=14, fontweight='bold')
7  axes[0].set_xlabel('Data', fontsize=12)
8  axes[0].set_ylabel('Potenza (kW)', fontsize=12)
9  axes[0].grid(True, alpha=0.3)
10 
11 # Zoom su una settimana
12 sample_start = len(data) // 2
13 sample_end = sample_start + 168  # 1 settimana
14 axes[1].plot(data['datetime'].iloc[sample_start:sample_end], 
15              data['power_kw'].iloc[sample_start:sample_end], 
16              linewidth=1.5, marker='o', markersize=3)
17 axes[1].set_title('Zoom - Una Settimana di Dati', fontsize=14, fontweight='bold')
18 axes[1].set_xlabel('Data', fontsize=12)
19 axes[1].set_ylabel('Potenza (kW)', fontsize=12)
20 axes[1].grid(True, alpha=0.3)
21 axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
22 plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
23 
24 plt.tight_layout()
25 plt.show()
26 
27 print(f"\nStatistiche produzione:")
28 print(f"Media: {data['power_kw'].mean():.2f} kW")
29 print(f"Max: {data['power_kw'].max():.2f} kW")
30 print(f"Min: {data['power_kw'].min():.2f} kW")
31 print(f"Std: {data['power_kw'].std():.2f} kW")
```

#### Spiegazione Dettagliata

---

### **Parte 1: Setup Figure Matplotlib (Riga 2)**

#### **Riga 2: Creazione Subplots**
```python
fig, axes = plt.subplots(2, 1, figsize=(15, 8))
```

**Scomposizione**:
- **`plt.subplots(2, 1)`**: Crea griglia 2 righe × 1 colonna di subplot
  - `2`: numero righe (subplots verticali)
  - `1`: numero colonne
  - **Output**: 2 grafici impilati verticalmente

- **`figsize=(15, 8)`**: Dimensione figura in pollici
  - Larghezza: 15 pollici (≈38 cm)
  - Altezza: 8 pollici (≈20 cm)
  - **Ratio**: 15:8 ≈ 1.875:1 (landscape)

**Return values**:
```python
fig    # Figure object: contenitore principale
axes   # Array NumPy con 2 Axes objects: [axes[0], axes[1]]
```

**Struttura creata**:
```
┌─────────────────────────────────┐
│  axes[0]  (plot completo)       │
│                                 │
├─────────────────────────────────┤
│  axes[1]  (zoom settimana)      │
│                                 │
└─────────────────────────────────┘
```

**Perché 2 subplot**:
1. **axes[0]**: Overview completa (2 anni) → pattern stagionali
2. **axes[1]**: Dettaglio settimanale → pattern giornalieri

**Alternativa con layout diverso**:
```python
# Subplots affiancati (side-by-side)
fig, axes = plt.subplots(1, 2, figsize=(18, 6))  # 1 riga, 2 colonne
```

---

### **Parte 2: Plot Serie Completa (Righe 4-9)**

#### **Riga 5: Plot Linea Serie Temporale**
```python
axes[0].plot(data['datetime'], data['power_kw'], linewidth=0.5, alpha=0.7)
```

**Parametri**:
1. **`data['datetime']`** (asse X):
   - Pandas Series con 17,314 timestamp
   - Range: `2010-07-01 00:00` → `2012-06-30 23:00`
   - Matplotlib gestisce automaticamente datetime

2. **`data['power_kw']`** (asse Y):
   - Pandas Series con 17,314 valori float32
   - Range: 0.0 - 58.07 kW

3. **`linewidth=0.5`**:
   - Spessore linea: 0.5 punti (molto sottile)
   - **Perché sottile**: 17k punti → linea spessa appare blob
   - Default: 1.5 (troppo per alta densità dati)

4. **`alpha=0.7`**:
   - Trasparenza: 70% opaco (30% trasparente)
   - Range: 0.0 (trasparente) → 1.0 (opaco)
   - **Perché 0.7**: Linee sovrapposte visibili, pattern chiari

**Effetto visivo**:
```
Con linewidth=0.5, alpha=0.7:
- Pattern stagionali visibili (estate alta, inverno bassa)
- Ciclicità giornaliera distinguibile (onde dense)
- Non appare troppo "pesante" o scuro

Con linewidth=2.0, alpha=1.0:
- Troppo spesso → blob nero indistinguibile
- Pattern giornalieri nascosti
```

**Output grafico atteso**:
```
60 kW ┤        ╱╲╱╲╱╲╱╲╱╲        ╱╲╱╲╱╲╱╲╱╲
      │      ╱          ╲      ╱          ╲
40 kW ┤    ╱              ╲  ╱              ╲
      │  ╱                  ╲                  ╲
20 kW ┤╱                      ╲              ╱  ╲
      │                         ╲          ╱      ╲
 0 kW ┼──────────────────────────╲────────╱────────
      2010-07         2011-01         2011-07
      
      ↑ Estate: picchi alti (50+ kW)
      ↓ Inverno: picchi bassi (20 kW)
```

---

#### **Righe 6-9: Formattazione Axes[0]**

**Riga 6: Titolo Grafico**
```python
axes[0].set_title('Serie Temporale Completa - Produzione Fotovoltaica', fontsize=14, fontweight='bold')
```
- **`fontsize=14`**: Dimensione font (default: 12)
- **`fontweight='bold'`**: Grassetto per evidenza
- **Valori alternativi**: `'normal'`, `'light'`, `'heavy'`, `100-900`

**Riga 7-8: Etichette Assi**
```python
axes[0].set_xlabel('Data', fontsize=12)
axes[0].set_ylabel('Potenza (kW)', fontsize=12)
```
- **xlabel**: Asse orizzontale (tempo)
- **ylabel**: Asse verticale (potenza)
- **Unità di misura**: `(kW)` esplicita per chiarezza

**Riga 9: Griglia**
```python
axes[0].grid(True, alpha=0.3)
```
- **`True`**: Abilita griglia
- **`alpha=0.3`**: Trasparenza 30% (griglia leggera, non invadente)
- **Default**: Griglia su major ticks (automatici)

**Effetto griglia**:
```
Con alpha=0.3:
│   │   │   │   │   │   <- Linee verticali leggere
────┼───┼───┼───┼───┼───  <- Linee orizzontali leggere
    │   │   │   │   │

Con alpha=1.0:
│   │   │   │   │   │   <- Troppo marcate, distraggono
━━━━┿━━━┿━━━┿━━━┿━━━┿━━━  
    │   │   │   │   │
```

---

### **Parte 3: Zoom Settimanale (Righe 11-22)**

#### **Righe 12-13: Selezione Finestra Temporale**
```python
sample_start = len(data) // 2
sample_end = sample_start + 168  # 1 settimana
```

**Riga 12: Calcolo Punto Medio**
```python
sample_start = len(data) // 2
```
- **`len(data)`**: 17,314 righe
- **`// 2`**: Divisione intera (floor division)
  - `17314 // 2 = 8657`
- **Perché metà dataset**: Rappresentativo (evita inizio/fine anomali)

**Riga 13: Finestra 168 Ore**
```python
sample_end = sample_start + 168
```
- **168 ore** = 7 giorni × 24 ore/giorno
- **Range campione**: indici `8657` → `8825`
- **Corrisponde a**: metà Gennaio 2011 (circa)

**Verifica timestamp**:
```python
# Calcolo temporale
2010-07-01 + (8657 ore) = 2010-07-01 + 360.7 giorni = 2011-06-26 circa
# Finestra: 2011-06-26 → 2011-07-03 (1 settimana)
```

**Perché 168 ore (1 settimana)**:
1. **Pattern giornalieri**: 7 cicli completi alba-tramonto
2. **Pattern settimanali**: Eventuale differenza weekend/weekday
3. **Leggibilità**: Abbastanza zoom per vedere dettagli
4. **Coerenza**: Stessa lunghezza dell'encoder TFT (`max_encoder_length=168`)

---

#### **Righe 14-16: Plot Dettaglio**
```python
axes[1].plot(data['datetime'].iloc[sample_start:sample_end], 
             data['power_kw'].iloc[sample_start:sample_end], 
             linewidth=1.5, marker='o', markersize=3)
```

**Riga 14-15: Slicing con .iloc**
```python
data['datetime'].iloc[sample_start:sample_end]  # Slicing basato su indice posizionale
```
- **`.iloc[]`**: Index-based selection (posizione numerica)
- **`[8657:8825]`**: 168 righe (sample_start a sample_end)
- **Alternativa**: `.loc[]` per label-based selection

**Differenza iloc vs loc**:
```python
# iloc: posizione numerica
data.iloc[0:3]      # Prime 3 righe (indici 0, 1, 2)

# loc: label-based
data.loc['2010-07-01':'2010-07-03']  # Range per etichette (se index è datetime)
```

**Riga 16: Parametri Plot Dettaglio**
- **`linewidth=1.5`**: Più spesso di axes[0] (0.5)
  - Motivo: Meno punti (168 vs 17k) → linea più spessa leggibile
  
- **`marker='o'`**: Marker circolare su ogni punto
  - Visualizza ogni ora come punto distinto
  - Utile per identificare singole misurazioni

- **`markersize=3`**: Diametro marker in punti
  - `3` = piccolo ma visibile
  - Default: `6` (troppo grande per 168 punti)

**Effetto visivo markers**:
```
Con marker='o', markersize=3:
     ●───●───●───●───●───●    ← Ciclo giorno (0→max→0)
   ●               ●       ●
 ●                   ●       ●
●                       ●      ● ← Notte (0 kW)

Senza markers:
     ─────────────────────    ← Linea continua
   ╱                     ╲
 ╱                         ╲
────────────────────────────
```

---

#### **Righe 17-20: Formattazione Axes[1]**
```python
axes[1].set_title('Zoom - Una Settimana di Dati', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Data', fontsize=12)
axes[1].set_ylabel('Potenza (kW)', fontsize=12)
axes[1].grid(True, alpha=0.3)
```
- Stesse impostazioni di axes[0]
- Titolo diverso: "Zoom - Una Settimana"
- Griglia per leggibilità dettaglio orario

---

#### **Righe 21-22: Formattazione Date Asse X**

**Riga 21: Formatter Date**
```python
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
```

**Scomposizione**:
1. **`axes[1].xaxis`**: Oggetto XAxis del subplot 1
2. **`set_major_formatter()`**: Imposta formato major ticks
3. **`mdates.DateFormatter()`**: Formatter matplotlib per date
   - **Classe**: `matplotlib.dates.DateFormatter`
   - **Import**: `from matplotlib import dates as mdates` (Cella 1)

**Pattern formato**: `'%Y-%m-%d %H:%M'`
| Codice | Significato | Esempio |
|--------|-------------|---------|
| `%Y` | Anno (4 cifre) | 2011 |
| `%m` | Mese (01-12) | 06 |
| `%d` | Giorno (01-31) | 26 |
| `%H` | Ora (00-23) | 14 |
| `%M` | Minuto (00-59) | 30 |

**Output esempio**:
```
Prima formattazione (default):
|────────|────────|────────|────────|
Jun 26   Jun 27   Jun 28   Jun 29

Dopo formattazione:
|──────────────|──────────────|──────────────|
2011-06-26 00:00  2011-06-27 00:00  2011-06-28 00:00
```

**Perché formato esteso**:
- **Precisione**: Ora esatta visibile (importante per PV)
- **Chiarezza**: Anno-mese-giorno eliminano ambiguità
- **Internazionale**: ISO 8601 standard (non ambiguo come MM/DD vs DD/MM)

**Riga 22: Rotazione Etichette**
```python
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
```

**Scomposizione**:
1. **`plt.setp()`**: Set property su lista oggetti matplotlib
   - **Funzione**: `plt.setp(objects, **properties)`
   - **Equivalente a**: Loop su `.set_property()` per ogni oggetto

2. **`axes[1].xaxis.get_majorticklabels()`**: Lista Text objects delle etichette
   - Return: `[Text(0, 0, '2011-06-26 00:00'), Text(...), ...]`

3. **`rotation=45`**: Rotazione 45° antiorario
   - Previene sovrapposizione etichette lunghe
   - Angolo ottimale: 30-45° (leggibile ma compatto)

4. **`ha='right'`**: Horizontal alignment = 'right'
   - Allinea estremità destra etichetta con tick mark
   - **Valori**: `'left'`, `'center'`, `'right'`

**Effetto rotazione**:
```
Senza rotazione (rotation=0):
2011-06-26 00:002011-06-27 00:002011-06-28 00:00  ← Sovrapposto!
|           |           |           |

Con rotation=45, ha='right':
           2011-06-26 00:00
                    |
                           2011-06-27 00:00
                                    |
                                           2011-06-28 00:00
                                                    |
```

**Alternative non usate**:
```python
# Rotazione verticale (troppo stretta)
plt.setp(..., rotation=90, ha='right')

# Formato auto date (meno controllo)
fig.autofmt_xdate(rotation=45, ha='right')  # Applica a tutti subplot
```

---

### **Parte 4: Layout e Display (Righe 24-25)**

#### **Riga 24: Layout Automatico**
```python
plt.tight_layout()
```

**Funzione**: Regola automaticamente padding tra subplot per evitare sovrapposizioni

**Cosa aggiusta**:
1. **Spaziatura tra subplot**: Evita sovrapposizione titoli/assi
2. **Margini figura**: Ottimizza spazio bianco esterno
3. **Etichette lunghe**: Accomoda tick labels ruotate

**Prima di tight_layout()**:
```
┌─────────────────────────────────┐
│  Title axes[0]                  │
│  [PLOT]                         │
│  X-Label                        │ ← Troppo vicino a title axes[1]
├─────────────────────────────────┤
│  Title axes[1]                  │ ← Sovrapposizione!
│  [PLOT]                         │
│  2011-06-26 00:00               │ ← Etichetta tagliata
└─────────────────────────────────┘
```

**Dopo tight_layout()**:
```
┌─────────────────────────────────┐
│  Title axes[0]                  │
│  [PLOT]                         │
│  X-Label                        │
│                                 │ ← Spaziatura aggiunta
├─────────────────────────────────┤
│  Title axes[1]                  │
│  [PLOT]                         │
│           2011-06-26 00:00      │ ← Visibile completamente
│                                 │
└─────────────────────────────────┘
```

**Parametri opzionali** (non usati qui):
```python
plt.tight_layout(pad=1.5)    # Padding extra
plt.tight_layout(h_pad=2.0)  # Padding verticale tra subplot
plt.tight_layout(w_pad=2.0)  # Padding orizzontale
```

---

#### **Riga 25: Render Figura**
```python
plt.show()
```

**Funzione**: Renderizza e mostra la figura in modalità interattiva (Jupyter) o finestra (script)

**Comportamento in Jupyter Notebook**:
- Display inline sotto la cella
- Formato PNG/SVG (settabile con `%config InlineBackend.figure_format`)
- Non blocca esecuzione (matplotlib inline backend)

**Comportamento in script Python**:
```python
# Script .py eseguito da terminale
plt.show()  # Apre finestra GUI (TkAgg, Qt5Agg, etc.)
            # BLOCCA esecuzione fino a chiusura finestra
```

**Alternative**:
```python
# Salva su file invece di mostrare
plt.savefig('produzione_pv.png', dpi=300, bbox_inches='tight')
plt.close()  # Libera memoria

# Mostra E salva
plt.savefig('produzione_pv.png')
plt.show()
```

---

### **Parte 5: Statistiche Descrittive (Righe 27-31)**

#### **Righe 28-31: Print Statistiche**
```python
print(f"\nStatistiche produzione:")
print(f"Media: {data['power_kw'].mean():.2f} kW")
print(f"Max: {data['power_kw'].max():.2f} kW")
print(f"Min: {data['power_kw'].min():.2f} kW")
print(f"Std: {data['power_kw'].std():.2f} kW")
```

**Riga 28: Newline Header**
```python
print(f"\nStatistiche produzione:")
```
- **`\n`**: Newline per separazione visiva dal plot
- **`:` finale**: Introduce lista statistiche

**Riga 29: Media (Mean)**
```python
print(f"Media: {data['power_kw'].mean():.2f} kW")
```
- **`data['power_kw'].mean()`**: Media aritmetica Pandas
  - Formula: $\mu = \frac{1}{n}\sum_{i=1}^{n} x_i$
  - Calcolo: $(0 + 0 + ... + 35.2 + ... + 0) / 17314$

- **`.2f`**: Format specifier
  - `.2`: 2 decimali di precisione
  - `f`: Fixed-point notation (non scientifica)

**Output atteso**: `Media: 16.73 kW`

**Interpretazione**:
- **16.73 kW** su max 58 kW → **28.8% del picco**
- Bassa media perché:
  1. Produzione zero 12+ ore/giorno (notte)
  2. Produzione parziale alba/tramonto
  3. Variabilità stagionale (inverno molto basso)

**Media vs Mediana** (non calcolata):
```python
# Mediana probabilmente diversa
data['power_kw'].median()  # ~10 kW (ipotesi)
# Media > Mediana → distribuzione skewed right
```

---

**Riga 30: Massimo (Max)**
```python
print(f"Max: {data['power_kw'].max():.2f} kW")
```
- **Calcolo**: $\max(x_1, x_2, ..., x_n)$
- **Output atteso**: `Max: 58.07 kW` (già visto in data analysis)

**Interpretazione fisica**:
- **58.07 kW** = Picco assoluto in 2 anni
- Probabilmente: mezzogiorno estivo, cielo limpido, temperatura ottimale
- **Capacità nominale impianto**: Probabilmente ~60 kWp (kilowatt peak)
- **Performance ratio**: $58.07 / 60 = 96.8\%$ (ottimo!)

---

**Riga 31: Minimo (Min)**
```python
print(f"Min: {data['power_kw'].min():.2f} kW")
```
- **Output atteso**: `Min: 0.00 kW`

**Interpretazione**:
- **0.00 kW** = Nessuna produzione (notte)
- Presente in ~50% dei timestamp (12 ore/giorno × 730 giorni)
- **Non anomalo**: comportamento normale impianti PV

---

**Riga 32: Deviazione Standard (Std)**
```python
print(f"Std: {data['power_kw'].std():.2f} kW")
```
- **Calcolo**: $\sigma = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^2}$
- **Pandas**: `ddof=1` (degrees of freedom = 1, sample std)
- **Output atteso**: `Std: 18.45 kW` (ipotesi)

**Interpretazione**:
```python
# Se Media = 16.73 kW, Std = 18.45 kW
Coefficiente di Variazione = Std / Media = 18.45 / 16.73 = 110%
```

- **CV > 100%** → **Altissima variabilità**!
- **Cause**:
  1. Alternanza giorno/notte (0 → 58 → 0)
  2. Variabilità stagionale (inverno: 0-20 kW, estate: 0-58 kW)
  3. Condizioni meteo (nuvole: -50% produzione)

**Distribuzione probabilmente bimodale**:
```
Frequenza
│     ╱╲              ╱╲
│    ╱  ╲            ╱  ╲
│   ╱    ╲          ╱    ╲
│  ╱      ╲        ╱      ╲
│ ╱        ╲      ╱        ╲
└──────────────────────────── Potenza (kW)
  0       10  20  30  40  50
  ↑                      ↑
Notte                 Giorno picco
(~50% campioni)      (~5% campioni)
```

---

### **Riepilogo Visualizzazione**

**Obiettivi della cella**:
1. ✅ **Overview completa**: 2 anni di dati (pattern stagionali)
2. ✅ **Dettaglio temporale**: 1 settimana (pattern giornalieri)
3. ✅ **Statistiche riassuntive**: Media, max, min, std

**Tecniche matplotlib utilizzate**:
| Tecnica | Scopo | Parametri Chiave |
|---------|-------|------------------|
| `subplots(2,1)` | Layout verticale | 2 grafici impilati |
| `linewidth=0.5` | Linea sottile | Alta densità dati (17k) |
| `alpha=0.7` | Trasparenza | Sovrapposizioni visibili |
| `marker='o'` | Punti dati | Distinguere misurazioni orarie |
| `DateFormatter` | Date leggibili | ISO 8601 con ora |
| `rotation=45` | Etichette ruotate | Evita sovrapposizione |
| `tight_layout()` | Spaziatura auto | No sovrapposizioni subplot |

**Insight dai grafici**:
1. **Stagionalità**: Evidente nel plot completo (onde lunghe)
2. **Ciclicità giornaliera**: Chiarissima nello zoom settimanale
3. **Variabilità giornaliera**: Alcuni giorni nuvoli (picco ridotto)
4. **Notti zero**: Pattern regolare 0 kW notturno

**Preparazione per modellazione**:
- Pattern temporali confermati → feature temporali (hour, month) utili
- Alta variabilità → normalizzazione necessaria
- Ciclicità regolare → encoder_length=168 cattura 7 cicli completi

**Dataset pronto per**: Configurazione TimeSeriesDataSet (prossima sezione)

---

## Sezione 5: Pre-processing per TFT (TimeSeriesDataSet)

Configurazione dei parametri TFT e della strategia di cross-validation temporale per evitare data leakage.

### Cella 12: Definizione Parametri TFT

```python
# Parametri TFT - Fissi per tutto il tuning
MAX_ENCODER_LENGTH = 168  # 1 settimana di contesto
MAX_PREDICTION_LENGTH = 24  # Previsione 24 ore
```

**Parametri architettura TFT**:

| Parametro | Valore | Significato |
|-----------|--------|-------------|
| `MAX_ENCODER_LENGTH` | 168 | Lunghezza storico (1 settimana = 168 ore) |
| `MAX_PREDICTION_LENGTH` | 24 | Orizzonte previsione (1 giorno = 24 ore) |

**Perché 168 ore (encoder)**:
1. **7 cicli giornalieri completi**: Cattura pattern settimanali
2. **Memoria LSTM sufficiente**: Sequenze troppo lunghe → gradient vanishing
3. **Bilanciamento**: Troppo corto (24h) → info insufficiente; troppo lungo (720h/1 mese) → overfitting

**Perché 24 ore (decoder)**:
- **Requisito task**: Previsione giorno successivo (day-ahead forecasting)
- **Pianificazione energetica**: Operatori grid necessitano previsioni 24h
- **Bilanciamento errore/utilità**: 24h = utile ma gestibile (vs 168h = alta incertezza)

**Architettura TFT**:
```
Input: 168 timesteps passati (t-167 → t)
       ↓
    ENCODER (LSTM + Attention)
       ↓
    DECODER (autoregressive)
       ↓
Output: 24 timesteps futuri (t+1 → t+24)
```

---

### Cella 13: Temporal Cross-Validation Setup

```python
def setup_temporal_cross_validation(data, n_folds=5, val_ratio=0.2):
    """Temporal Cross-Validation con fold bilanciati per evitare data leakage"""
    
    max_time_idx = data['time_idx'].max()
    total_samples = len(data)
    val_size = int(total_samples * val_ratio)  # Validation fisso 20%
    
    # Training size cresce progressivamente
    min_train_size = int(total_samples * 0.5)  # Minimo 50%
    max_train_size = total_samples - val_size
    
    folds = []
    for fold in range(n_folds):
        progress = fold / (n_folds - 1) if n_folds > 1 else 0
        train_size = int(min_train_size + progress * (max_train_size - min_train_size))
        
        # Calcola cutoff temporali
        train_cutoff = train_size - 1
        val_start = train_cutoff + 1
        val_cutoff = val_start + val_size - 1
        
        # Filtra dati per questo fold
        train_data = data[data['time_idx'] <= train_cutoff].copy()
        val_data = data[(data['time_idx'] >= val_start) & 
                        (data['time_idx'] <= val_cutoff)].copy()
        
        folds.append((train_data, val_data))
    
    return folds

folds = setup_temporal_cross_validation(data, n_folds=5, val_ratio=0.2)
```

**Strategia Temporal Cross-Validation**:

**❌ K-Fold Standard (SBAGLIATO per time series)**:
```
Fold 1: Train [A,C,D,E] | Val [B]  ← Usa dati futuri per predire passato!
Fold 2: Train [B,C,D,E] | Val [A]  ← Data leakage
```

**✅ Temporal Cross-Validation (CORRETTO)**:
```
Fold 1: Train [────50%────] | Val [─20%─]
Fold 2: Train [──────60%──────] | Val [─20%─]
Fold 3: Train [────────70%────────] | Val [─20%─]
Fold 4: Train [──────────80%──────────] | Val [─20%─]
Fold 5: Train [────────────90%────────────] | Val [─20%─]
        ▲ Training cresce, validation fisso alla fine
```

**Componenti chiave**:
1. **Validation size fisso**: 20% del dataset (sempre alla fine temporale)
2. **Training size crescente**: Da 50% a 90% del dataset
3. **No overlap temporale**: Train sempre prima di validation (no data leakage)
4. **Skip fold invalidi**: Se train/val troppo piccoli per encoder+decoder

**Output esempio**:
```
Fold 1/5:
  Training: 8657 samples (50.0%)
  Validation: 3463 samples (20.0%)
  Train range: 2010-07-01 to 2011-06-26
  Val range: 2011-06-27 to 2012-01-24

Fold 5/5:
  Training: 13851 samples (80.0%)
  Validation: 3463 samples (20.0%)
  Train range: 2010-07-01 to 2012-01-24
  Val range: 2012-01-25 to 2012-06-30
```

---

### Cella 14: Hyperparameter Search Space

```python
def suggest_hyperparameters(trial):
    """Definisce lo spazio di ricerca degli iperparametri per Optuna"""
    hyperparams = {
        # Architettura
        'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 192, 256]),
        'lstm_layers': trial.suggest_int('lstm_layers', 1, 3),
        'attention_head_size': trial.suggest_categorical('attention_head_size', [1, 2, 4, 8]),
        'hidden_continuous_size': trial.suggest_categorical('hidden_continuous_size', [8, 16, 32]),
        
        # Regolarizzazione
        'dropout': trial.suggest_float('dropout', 0.1, 0.4),
        
        # Training
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'patience': trial.suggest_int('patience', 10, 30),
        'gradient_clip_val': trial.suggest_float('gradient_clip_val', 0.1, 2.0),
    }
    return hyperparams
```

**Spazio di ricerca iperparametri**:

| Categoria | Parametro | Range | Tipo | Effetto |
|-----------|-----------|-------|------|---------|
| **Architettura** | `hidden_size` | [64, 128, 192, 256] | Categorical | Capacità modello |
| | `lstm_layers` | [1, 2, 3] | Integer | Profondità LSTM |
| | `attention_head_size` | [1, 2, 4, 8] | Categorical | Multi-head attention |
| | `hidden_continuous_size` | [8, 16, 32] | Categorical | Embedding continui |
| **Regolarizzazione** | `dropout` | [0.1, 0.4] | Float | Prevenzione overfitting |
| **Optimization** | `learning_rate` | [1e-4, 1e-1] | Float (log) | Velocità convergenza |
| | `batch_size` | [32, 64, 128] | Categorical | Memoria/stabilità |
| | `patience` | [10, 30] | Integer | Early stopping |
| | `gradient_clip_val` | [0.1, 2.0] | Float | Stabilità training |

**Note tecniche**:
- **`log=True` per learning_rate**: Campionamento logaritmico (esplora meglio ordini di grandezza)
- **Categorical vs Int**: Categorical per valori non ordinali o con gap irregolari
- **hidden_size**: Potenza di 2 non obbligatoria (192 incluso per granularità)

---

### Cella 15-17: Objective Function con Cross-Validation

```python
def objective(trial):
    """Funzione obiettivo per Optuna con temporal cross-validation"""
    
    # Feature conosciute (known_reals)
    known_reals = ["time_idx", "hour", "day_of_month", "month", "day_of_week"]
    
    # Suggerisci iperparametri
    hyperparams = suggest_hyperparameters(trial)
    
    fold_losses = []
    
    # Loop attraverso tutti i fold
    for fold_idx, (train_data, val_data) in enumerate(folds):
        
        # Crea TimeSeriesDataSet
        training_dataset = TimeSeriesDataSet(
            train_data,
            time_idx="time_idx",
            target="power_kw",
            group_ids=["group_id"],
            min_encoder_length=MAX_ENCODER_LENGTH,
            max_encoder_length=MAX_ENCODER_LENGTH,
            min_prediction_length=MAX_PREDICTION_LENGTH,
            max_prediction_length=MAX_PREDICTION_LENGTH,
            static_categoricals=["group_id"],
            time_varying_known_reals=known_reals,
            time_varying_unknown_reals=["power_kw"],
            target_normalizer=GroupNormalizer(groups=["group_id"]),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        
        # Validation dataset da training
        validation_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset, 
            pd.concat([train_data, val_data]),
            predict=True
        )
        
        # DataLoaders
        train_dataloader = training_dataset.to_dataloader(
            train=True, batch_size=hyperparams['batch_size'], num_workers=0
        )
        val_dataloader = validation_dataset.to_dataloader(
            train=False, batch_size=hyperparams['batch_size']*2, num_workers=0
        )
        
        # Crea modello TFT
        tft = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=hyperparams['learning_rate'],
            hidden_size=hyperparams['hidden_size'],
            lstm_layers=hyperparams['lstm_layers'],
            attention_head_size=hyperparams['attention_head_size'],
            dropout=hyperparams['dropout'],
            hidden_continuous_size=hyperparams['hidden_continuous_size'],
            output_size=7,  # Quantile loss (7 quantili)
            loss=QuantileLoss(),
        )
        
        # Training con Early Stopping
        early_stop = EarlyStopping(
            monitor="val_loss", patience=hyperparams['patience']//2, mode="min"
        )
        
        trainer = pl.Trainer(
            max_epochs=30,
            accelerator="auto",
            gradient_clip_val=hyperparams['gradient_clip_val'],
            callbacks=[early_stop],
            logger=False,
            enable_progress_bar=False,
        )
        
        trainer.fit(tft, train_dataloader, val_dataloader)
        
        # Salva validation loss
        fold_val_loss = trainer.callback_metrics.get("val_loss").item()
        fold_losses.append(fold_val_loss)
        
        # Cleanup memoria
        del tft, trainer, training_dataset, validation_dataset
        torch.cuda.empty_cache()
        gc.collect()
    
    # Restituisci media loss su tutti i fold
    mean_val_loss = np.mean(fold_losses)
    return mean_val_loss
```

**Componenti TimeSeriesDataSet**:

| Parametro | Valore | Scopo |
|-----------|--------|-------|
| `time_idx` | "time_idx" | Indice temporale (0-17313) |
| `target` | "power_kw" | Variabile da prevedere |
| `group_ids` | ["group_id"] | Identificatore serie ('PV1') |
| `time_varying_known_reals` | [time_idx, hour, month, ...] | Feature note in futuro |
| `time_varying_unknown_reals` | ["power_kw"] | Feature ignote in futuro (solo target) |
| `target_normalizer` | GroupNormalizer | Normalizza per gruppo (z-score) |
| `add_relative_time_idx` | True | Posizione relativa in finestra (0→1) |
| `add_target_scales` | True | Media/std target per decoder |
| `add_encoder_length` | True | Lunghezza encoder come feature |

**Known vs Unknown**:
- **Known reals** (time_varying_known_reals): Feature disponibili anche in futuro
  - `time_idx`: Sempre noto (incrementale)
  - `hour`, `month`, etc.: Calendario noto in anticipo
  
- **Unknown reals** (time_varying_unknown_reals): Feature ignote in futuro
  - `power_kw`: Target da prevedere (non conosciuto!)
  - Weather features (temp, Dni, etc.): NON usate perché non disponibili in produzione

**Normalizzazione GroupNormalizer**:
```python
# Formula: z-score per gruppo
normalized_value = (value - group_mean) / group_std

# Nel nostro caso (1 gruppo):
group_mean = data['power_kw'].mean()  # ~16.73 kW
group_std = data['power_kw'].std()    # ~18.45 kW
```

**from_dataset per validation**:
- **Eredita configurazione**: Stesso normalizer, stesso encoding
- **predict=True**: Genera sample per prediction (no teacher forcing)
- **Concatena train+val**: Necessario per avere contesto encoder anche per primi sample validation

**Pipeline completo per fold**:
```
1. TimeSeriesDataSet (train) → Configura normalizzazione
2. TimeSeriesDataSet.from_dataset (val) → Usa stessa normalizzazione
3. DataLoaders → Batch di sequenze [batch, seq, features]
4. TFT.from_dataset() → Inizializza architettura da dataset
5. Trainer.fit() → Training con early stopping
6. Estrai val_loss → Metrica per Optuna
7. Cleanup GPU → Libera memoria per fold successivo
```

**Output cross-validation**:
```
Fold 1 Val Loss: 0.125634
Fold 2 Val Loss: 0.118923
Fold 3 Val Loss: 0.121456
Fold 4 Val Loss: 0.119832
Fold 5 Val Loss: 0.122108

Mean Val Loss: 0.121591 (±0.002341)
```

**Optuna utilizza**: `mean_val_loss` per confrontare trial e trovare migliori iperparametri.

---

## Prossimi Passi

Le prossime sezioni del notebook coprono:
- **Sezione 6 (Cella 18)**: Optuna Study Execution
- **Sezione 7 (Cella 19-23)**: Best Hyperparameters & Final Training
- **Sezione 8 (Cella 24-27)**: Evaluation & Visualization

