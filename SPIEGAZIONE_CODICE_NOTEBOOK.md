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
20     print("\n⚠️  GPU non disponibile. Possibili cause:")
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

## Prossimi Passi

Le prossime sezioni del notebook coprono:
- **Sezione 3 (Celle 7-9)**: Data Analysis & Missing Values
- **Sezione 4 (Celle 10-11)**: Feature Engineering
- **Sezione 5 (Celle 12-14)**: TimeSeriesDataSet Configuration

**Vuoi che continui con le celle 7-9?**

