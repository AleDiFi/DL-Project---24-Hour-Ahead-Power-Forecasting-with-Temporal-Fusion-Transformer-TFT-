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

## Fine Spiegazione Prime 3 Celle
