# data_pipeline_pretrained.py

import numpy as np
import tensorflow as tf
import tifffile
from tensorflow.keras.applications import (
    resnet50, vgg16, inception_v3
)

AUTOTUNE = tf.data.AUTOTUNE

# ---- Base I/O (riuso dal baseline; duplico per autosufficienza del modulo) ----
def load_ms_13(path):
    arr = tifffile.imread(path).astype(np.float32)  # (64,64,13) in EuroSAT_MS
    if arr.ndim != 3 or arr.shape[-1] != 13:
        raise ValueError(f"Atteso (H,W,13), ottenuto {arr.shape} per {path}")
    return arr

def normalize_ms_13(arr, mean_per_band, std_per_band):
    return (arr - mean_per_band[None, None, :]) / std_per_band[None, None, :]

# ---- Utility: preprocess_input per modello ----
def get_preprocess_fn(model_name):
    name = model_name.lower()
    if name == "resnet50":
        return resnet50.preprocess_input
    elif name == "vgg16":
        return vgg16.preprocess_input
    elif name == "inception_v3":
        return inception_v3.preprocess_input
    else:
        # fallback neutro: nessuna trasformazione
        return lambda x: x

# ---- Branch 1: RGB_only (B4,B3,B2) + preprocess_input ImageNet ----
def _py_rgb_only(path_bytes, label_int, rgb_idx, input_size, model_name):
    path = path_bytes.numpy().decode("utf-8")                 # tf.string -> bytes -> str
    label = int(label_int.numpy())                            # tf.int32  -> int
    rgb_idx = tuple(int(i) for i in rgb_idx.numpy().tolist()) # tf.Tensor -> tuple di int
    input_size = tuple(int(i) for i in input_size.numpy().tolist())  # tf.Tensor -> (H,W)
    model_name = model_name.numpy().decode("utf-8")           # tf.string -> str

    arr  = load_ms_13(path)                                  # (64,64,13)
    rgb  = arr[..., rgb_idx]                                 # (64,64,3) B4,B3,B2
    # resize a input_size (H,W), mantenendo float32
    rgb_res = tf.image.resize(rgb, input_size, method="bilinear").numpy()
    # preprocess "alla ImageNet" del backbone scelto, non usiamo nostra mean/std!!
    pp = get_preprocess_fn(model_name)
    # applica preprocess_input specifico del backbone
    rgb_pre = pp(rgb_res)                                    # shape (H,W,3)
    return rgb_pre.astype(np.float32), np.int32(label_int)

def tf_map_rgb_only(path_tensor, label_tensor, rgb_idx, input_size, model_name):
    img, y = tf.py_function(
        func=_py_rgb_only,
        inp=[path_tensor, label_tensor, tf.constant(rgb_idx, dtype=tf.int32), tf.constant(input_size, dtype=tf.int32), tf.constant(model_name)],
        Tout=(tf.float32, tf.int32)
    )
    img.set_shape((input_size[0], input_size[1], 3))
    y.set_shape(())
    return img, y

# ---- Branch 2: 13 bande normalizzate (per 1x1_conv_to_RGB a livello di modello) ----
def _py_ms13_norm(path_bytes, label_int, mean13, std13, input_size):
    path = path_bytes.numpy().decode("utf-8")
    label = int(label_int.numpy())
    mean13 = mean13.numpy()                                   # tf.Tensor -> np.ndarray (13,)
    std13  = std13.numpy()
    input_size = tuple(int(i) for i in input_size.numpy().tolist())

    arr  = load_ms_13(path)                                  # (64,64,13)
    arr  = normalize_ms_13(arr, mean13, std13)               # standardizza per banda
    arr_res = tf.image.resize(arr, input_size, method="bilinear").numpy()  # (H,W,13)
    return arr_res.astype(np.float32), np.int32(label_int)

def tf_map_ms13_norm(path_tensor, label_tensor, mean13, std13, input_size):
    img, y = tf.py_function(
        func=_py_ms13_norm,
        inp=[path_tensor, label_tensor, tf.constant(mean13, dtype=tf.float32), tf.constant(std13, dtype=tf.float32), tf.constant(input_size, dtype=tf.int32)],
        Tout=(tf.float32, tf.int32)
    )
    img.set_shape((input_size[0], input_size[1], 13))
    y.set_shape(())
    return img, y

# ---- Dataset builder (unico punto di ingresso) ----
def make_dataset_pretrained(file_list,
                            label2id,
                            batch_size,
                            augment,
                            seed,
                            input_size,
                            channel_strategy,
                            model_name,
                            rgb_idx=(3,2,1),
                            mean13=None,
                            std13=None):
    """
    Crea un tf.data.Dataset pronto per model.fit, in base alla strategia canali.

    file_list       : lista [(filepath, label_str), ...]
    label2id        : dict {label_str -> int}
    batch_size      : int
    augment         : bool (solo train)
    seed            : int
    input_size      : (H,W)
    channel_strategy: "RGB_only" | "1x1_conv_to_RGB"
    model_name      : "resnet50" | "vgg16" | "inception_v3" | ...
    rgb_idx         : tuple/list di 3 indici per B4,B3,B2
    mean13,std13    : necessari se channel_strategy = "1x1_conv_to_RGB"
    """
    # separo liste semplici
    filepaths = [fp for fp, _ in file_list]
    labels    = [label2id[lbl] for _, lbl in file_list]

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    # mappa di preprocessing
    if channel_strategy == "RGB_only":
        ds = ds.map(
            lambda p, y: tf_map_rgb_only(p, y, rgb_idx, input_size, model_name),
            num_parallel_calls=AUTOTUNE
        )
    elif channel_strategy == "1x1_conv_to_RGB":
        if mean13 is None or std13 is None:
            raise ValueError("Per '1x1_conv_to_RGB' servono mean13 e std13 (dal train).")
        ds = ds.map(
            lambda p, y: tf_map_ms13_norm(p, y, mean13, std13, input_size),
            num_parallel_calls=AUTOTUNE
        )
    else:
        raise ValueError(f"channel_strategy non supportata: {channel_strategy}")

    # augmentation solo se richiesto (sul train tipicamente)
    if augment:
        ds = ds.map(lambda x, y: (tf.image.random_flip_left_right(x), y),
                    num_parallel_calls=AUTOTUNE)

    # batch + prefetch
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds