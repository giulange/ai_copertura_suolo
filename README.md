# 🌍 Verso l’AI: Reti Neurali per la Mappatura della Copertura del Suolo

![Locandina seminario](./virtualbackground_langella.jpg)

## 📌 Evento
**Estate GIS 2025**  
**Titolo:** Verso l’AI: Reti neurali per la mappatura della copertura del suolo  
**Relatore:** Giuliano Langella  
**Data:** 10 settembre 2025  
**Orario:** 17.30 – 19.30  

## 📂 Struttura del repository
- `README.md` → questa pagina introduttiva.  
- `requirements.txt` → librerie Python necessarie.  
- `docker-compose.yml` e `run.sh` → ambiente esecutivo.  
- `supclassann/` → materiali didattici del seminario:  
  - Notebook Jupyter interattivi.  
  - Script Python per pipeline di dati.  
  - Documentazione (`.md`).  
  - Artefatti e output generati durante le sessioni.  

## 🎯 Obiettivi
Il repository raccoglie i materiali del seminario, con esempi pratici di:
- **Addestramento di reti neurali convoluzionali (CNN)** su dati di telerilevamento.  
- **Transfer learning** con modelli pre-addestrati (ResNet-50, VGG16, Inception).  
- **Pipeline dati** per EuroSAT e dataset multispettrali.  

## 🚀 Come iniziare
### 1. Clonare la repository:  
   ```bash
   git clone https://github.com/<tuo-utente>/ai_copertura_suolo.git
   cd ai_copertura_suolo
   ```
### 2.	Creare l'ambiente di lavoro

#### Opzione A: iunstallare i requisiti Python
   ```bash
   pip install -r requirements.txt
   ```

#### Opzione B (consigliata): usare un contenitore Docker

1. **Installare Docker Desktop** (include già Docker Engine e Compose):
   - **Windows**: scaricare da [Docker Desktop per Windows](https://docs.docker.com/desktop/install/windows/).  
     > ⚠️ Richiede **WSL2** attivato (Windows Subsystem for Linux). Durante l’installazione Docker Desktop ti guiderà a configurarlo.  
   - **macOS**: scaricare da [Docker Desktop per Mac](https://docs.docker.com/desktop/install/mac/).  
   - **Linux (Ubuntu/Debian)**: installare direttamente da pacchetto o repository ufficiale:  
     ```bash
     sudo apt-get update
     sudo apt-get install docker.io docker-compose-plugin
     sudo systemctl enable docker
     sudo systemctl start docker
     ```

2. **Verificare l’installazione**:
   ```bash
   docker --version
   docker compose version
   ```

3.	**Avviare l’ambiente del seminario** (esegue build ed esecuzione del container):

   ```bash
   ./run.sh
   ```

> Con Docker l’ambiente è isolato e già configurato: non serve installare manualmente le librerie Python.

👉 Su **Windows** lo script `run.sh` può essere eseguito dal **WSL2 terminal** (Ubuntu, Debian, ecc.), oppure da **Git Bash**.  
Se gli studenti non hanno familiarità con la shell, possono eseguire manualmente i comandi contenuti in `run.sh`.

## 📜 Licenza

Questo repository è distribuito con licenza **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.  
Puoi riutilizzare, modificare e condividere i materiali a condizione di:  

- **Attribuire** correttamente la fonte e l’autore.  
- **Non utilizzare** i materiali per scopi commerciali.  

Per i dettagli completi: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).