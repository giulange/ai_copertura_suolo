# 📘 Materiali del Seminario: Verso l’AI per la Copertura del Suolo
![Locandina seminario](../virtualbackground_langella.jpg)

Questa cartella contiene i **materiali didattici** del seminario *Verso l’AI: Reti neurali per la mappatura della copertura del suolo*.

## 📂 Contenuto

- **01_Introduzione.ipynb** → introduzione teorica e pratica alle reti neurali convoluzionali (CNN).  
- **02_eSat_custom_CNN.ipynb** → implementazione di una CNN personalizzata su EuroSAT.  
- **03_eSAT_pretrained_models.ipynb** → utilizzo di modelli pre-addestrati (ResNet-50, VGG16, Inception) con transfer learning.  
- **data_pipeline_pretrained.py** → libreria per la gestione dei dati multispettrali e preprocessing per i modelli.  
- **data_pipeline_pretrained.md** → documentazione della libreria dati.  
- **artifacts/** → figure, grafici e materiali prodotti durante le esercitazioni.  
- **artwork/** → immagini di supporto usate nei notebook.  
- **outputs/** → output dei modelli addestrati (log, metriche, modelli intermedi).  
- **outputs_pretrained/** → pesi e risultati specifici del transfer learning.  
- **split_lists/** → liste di train/val/test per garantire la tracciabilità degli esperimenti.  

## 🎯 Obiettivi didattici

- Capire le differenze tra una **CNN personalizzata** e una **rete pre-addestrata**.  
- Applicare il **transfer learning** a dati satellitari (EuroSAT).  
- Confrontare diverse strategie di preprocessing (RGB-only, 1x1 conv per multispettrali).  
- Eseguire un addestramento in **2 fasi** (solo testa → fine-tuning).  

## 📥 Dataset EuroSAT_MS

I notebook forniti si basano sul dataset **EuroSAT multispettrale (EuroSAT_MS)**.  
Per motivi di licenza e dimensioni, **il dataset non è incluso in questa repository**.

### Come preparare i dati
1. Scaricare l’archivio `.zip` di EuroSAT_MS dal sito ufficiale (o fonte indicata dal docente).  
2. Estrarre l’archivio localmente.  
3. Copiare tutte le **cartelle delle classi** (es. `AnnualCrop/`, `Forest/`, `Pasture/`, ecc.) dentro la cartella EuroSAT_MS.

## 🚀 Come utilizzare

Aprire i notebook in ordine numerico (`01_`, `02_`, `03_`) per seguire il percorso formativo.  
Per l’esecuzione si raccomanda l’uso di un ambiente Python configurato con i requisiti del repository principale o il contenitore Docker.

---

✍️ **Nota**: Questo materiale è pensato per studenti e ricercatori che vogliono sperimentare con reti neurali applicate alla classificazione della copertura del suolo.  
