# Documentazione — `data_pipeline_pretrained.py`

> Libreria di utilità per costruire pipeline `tf.data` adatte a **backbone pre-addestrati** (VGG16, ResNet-50, InceptionV3) a partire da patch **EuroSAT multispettrali (13 bande)**.

---

## Costanti

### `AUTOTUNE`
- Alias: `tf.data.AUTOTUNE`.
- Fa sì che TensorFlow scelga automaticamente il grado di parallelismo in `map()/prefetch()`.

---

## Funzioni di I/O di base (riuso dal baseline)

### `load_ms_13(path) -> np.ndarray`
**Scopo:** legge un GeoTIFF multispettrale e restituisce un **array float32 (H,W,13)**.  
**Argomenti**
- `path` *(str)*: percorso del file `.tif`.  
**Ritorna**
- `arr` *(np.ndarray, float32, shape (H,W,13))*.

**Note**
- Lancia `ValueError` se le dimensioni non sono (H,W,13).

---

### `normalize_ms_13(arr, mean_per_band, std_per_band) -> np.ndarray`
**Scopo:** **standardizza per banda**: `(arr - mean) / std`.  
**Argomenti**
- `arr` *(np.ndarray, (H,W,13))*  
- `mean_per_band` *(np.ndarray, (13,))*  
- `std_per_band` *(np.ndarray, (13,))*  
**Ritorna**
- `arr_norm` *(np.ndarray, float32, (H,W,13))*.

**Quando usarla**
- Strategia **`1x1_conv_to_RGB`** (non `RGB_only`).

---

## Preprocess “alla ImageNet”

### `get_preprocess_fn(model_name) -> callable`
**Scopo:** restituisce la funzione `preprocess_input` coerente con il backbone.  
**Argomenti**
- `model_name` *(str)*: `"resnet50" | "vgg16" | "inception_v3" | ...`  
**Ritorna**
- Funzione che applica il **preprocess specifico** (mean-centering BGR per VGG/ResNet; scaling in `[-1,1]` per Inception).  
**Fallback**
- Se il nome non è riconosciuto, ritorna una funzione identità (`lambda x: x`).

---

## Mappature per `tf.data` — strategia **RGB_only**

> Selezioniamo le bande **B4,B3,B2** → ridimensioniamo → **preprocess Input** del backbone → shape `(H,W,3)`.

### `_py_rgb_only(path_bytes, label_int, rgb_idx, input_size, model_name) -> (np.ndarray, np.int32)`
**Scopo:** funzione “**Python puro**” chiamata da `tf.py_function`.  
**Argomenti**
- `path_bytes` *(bytes Tensor)*: percorso in bytes da convertire in `str`.  
- `label_int` *(int)*: etichetta già intera.  
- `rgb_idx` *(sequence di 3 int)*: indici delle bande RGB (es. `[3,2,1]`).  
- `input_size` *(tuple (H,W))*: dimensione desiderata (224×224 o 299×299).  
- `model_name` *(str)*: backbone (“resnet50”, “vgg16”, “inception_v3”).  
**Ritorna**
- `img` *(np.ndarray, float32, (H,W,3))* preprocessata per il backbone.  
- `y` *(np.int32)*: etichetta.

> **Privata**: pensata per essere usata tramite la wrapper TensorFlow qui sotto.

### `tf_map_rgb_only(path_tensor, label_tensor, rgb_idx, input_size, model_name) -> (tf.Tensor, tf.Tensor)`
**Scopo:** wrapper **TensorFlow** che usa `_py_rgb_only` via `tf.py_function` e **imposta la shape**.  
**Argomenti**
- `path_tensor` *(tf.string)*, `label_tensor` *(tf.int32)*  
- `rgb_idx` *(list/tuple di 3 int)*  
- `input_size` *(tuple (H,W))*  
- `model_name` *(str)*  
**Ritorna**
- `img` *(tf.float32, shape fissa (H,W,3))*  
- `y` *(tf.int32, shape `()`)*

---

## Mappature per `tf.data` — strategia **1x1_conv_to_RGB**

> **Standardizziamo** tutte le **13 bande** con `mean/std` del **train** → resize → shape `(H,W,13)`  
> (la **proiezione 13→3** avverrà nel **modello** con una `Conv2D(3, kernel=1)`).

### `_py_ms13_norm(path_bytes, label_int, mean13, std13, input_size) -> (np.ndarray, np.int32)`
**Scopo:** funzione “Python puro” per caricare e **standardizzare** (13 bande).  
**Argomenti**
- `path_bytes` *(bytes Tensor)*, `label_int` *(int)*  
- `mean13`, `std13` *(np.ndarray, (13,))*  
- `input_size` *(tuple (H,W))*  
**Ritorna**
- `img` *(np.ndarray, float32, (H,W,13))* standardizzata.  
- `y` *(np.int32)*.

> **Privata**: usala tramite il wrapper TF.

### `tf_map_ms13_norm(path_tensor, label_tensor, mean13, std13, input_size) -> (tf.Tensor, tf.Tensor)`
**Scopo:** wrapper TensorFlow di `_py_ms13_norm` con **shape set**.  
**Argomenti**
- `path_tensor` *(tf.string)*, `label_tensor` *(tf.int32)*  
- `mean13`, `std13` *(tf.float32, (13,))*  
- `input_size` *(tuple (H,W))*  
**Ritorna**
- `img` *(tf.float32, (H,W,13))*  
- `y` *(tf.int32, ( ))*

---

## Costruttore di dataset

### `make_dataset_pretrained(file_list, label2id, batch_size, augment, seed, input_size, channel_strategy, model_name, rgb_idx=(3,2,1), mean13=None, std13=None) -> tf.data.Dataset`
**Scopo:** crea un `tf.data.Dataset` **pronto per `model.fit`** in base alla strategia canali.

**Argomenti (più importanti)**
- `file_list` *(list of (path:str, label:str))*: righe dei CSV.  
- `label2id` *(dict)*: mapping classe→intero coerente con il baseline.  
- `batch_size` *(int)*, `augment` *(bool)*, `seed` *(int)*  
- `input_size` *(tuple (H,W))*: 224×224 (VGG/ResNet) o 299×299 (Inception).  
- `channel_strategy` *(str)*: `"RGB_only"` **o** `"1x1_conv_to_RGB"`.  
- `model_name` *(str)*: per selezionare il giusto `preprocess_input` (solo RGB_only).  
- `rgb_idx` *(3 int)*: indici delle bande (default: `[3,2,1]` → B4,B3,B2).  
- `mean13`, `std13` *(np.ndarray (13,))*: **obbligatori** se `"1x1_conv_to_RGB"`.

**Comportamento**
- Crea il dataset da liste Python → `from_tensor_slices`.  
- **`map`**:
  - se `"RGB_only"` → `tf_map_rgb_only` (selezione B4,B3,B2 + preprocess ImageNet).  
  - se `"1x1_conv_to_RGB"` → `tf_map_ms13_norm` (standardizzazione 13 bande).  
- **Augmentation** *(solo se `augment=True`)*: `random_flip_left_right`.  
- **`batch`** + **`prefetch(AUTOTUNE)`**.

**Ritorna**
- `tf.data.Dataset` in cui **ogni elemento** è:
  - immagini: `tf.float32` shape `(BATCH_SIZE, H, W, C)` con `C=3` o `13`
  - etichette: `tf.int32` shape `(BATCH_SIZE,)`.

---

## Esempi d’uso (nel notebook)

### A) Strategia `RGB_only` (consigliata per partire)
```python
train_ds = make_dataset_pretrained(
    file_list=train_list,
    label2id=label2id,
    batch_size=BATCH_SIZE,
    augment=AUGMENT,         # True solo sul train
    seed=SEED,
    input_size=INPUT_SIZE,   # 224x224 (VGG/ResNet) o 299x299 (Inception)
    channel_strategy="RGB_only",
    model_name=MODEL_NAME,   # "resnet50" | "vgg16" | "inception_v3"
    rgb_idx=[3,2,1]
)
```

### B) Strategia 1x1_conv_to_RGB
```python
# mean_per_band, std_per_band: calcolati sul SOLO train (dal baseline)
train_ds = make_dataset_pretrained(
    file_list=train_list,
    label2id=label2id,
    batch_size=BATCH_SIZE,
    augment=AUGMENT,
    seed=SEED,
    input_size=INPUT_SIZE,
    channel_strategy="1x1_conv_to_RGB",
    model_name=MODEL_NAME,
    mean13=mean_per_band,
    std13=std_per_band
)
# Nota: nel MODELLO aggiungerai una Conv2D(3, kernel_size=1) prima del backbone.
```