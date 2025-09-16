# %% [markdown]
# # CEAL – TensorFlow + MobileNetV2 (Notebook-Style .py) mit Cosine-Annealing, Pseudolabel-Logging & Plots
#
# - Cosine-Decay **ohne** Restarts (global über Warmup + alle Iterationen)
# - Strengere Pseudolabel-Gates (δ + min confidence + min margin)
# - CSV `correct` als `true`/`false`
# - **NEU:** Plots am Ende: Test-Acc über Iterationen, Pseudolabel-Acc, Labeled/Unlabeled-Verlauf,
#   High-Confidence pro Iteration, Trainingskurven (Loss/Accuracy je Epoche), LR-Schedule über Batches
#
# -- Vollständiger Code beginnt hier --

import os, math, random, json, glob, csv
from typing import List, Tuple, Dict, Optional
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Für Plots
import matplotlib.pyplot as plt

CONFIG = {
    "DATA_ROOT": r"C:\Users\smart.farming\Desktop\TomBA\Bilddaten\CEAL\ceal_split",
    "INIT_LABELED_DIR": "initial_data",
    "UNLABELED_DIR": "unlabeled_pool",
    "VAL_DIR": "val",
    "TEST_DIR": "test",
    "IMG_SIZE": 224,
    "BATCH_SIZE": 32,
    "EXPECT_NUM_CLASSES": 5,
    "ITERS": 20,
    "K": 400,
    "STRATEGY": "entropy",
    "DELTA0": 0.18,
    "DECAY": 0.02,
    "NORMALIZE_ENTROPY": True,
    "PSEUDO_MIN_CONF": 0.90,
    "PSEUDO_MIN_MARGIN": 0.05,
    "WARMUP_EPOCHS": 2,
    "EPOCHS_PER_ITER": 2,
    "USE_COSINE": True,
    "COSINE_INITIAL_LR": 1e-3,
    "COSINE_FINAL_LR_RATIO": 0.05,
    "WEIGHT_DECAY": 1e-5,
    "WEIGHTS": "imagenet",
    "BASE_TRAINABLE_AFTER": 2,
    "CSV_DIR": "ceal_logs",
    "PLOTS_DIR": "ceal_plots",
    "SEED": 123,
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

def seed_all(seed=123):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def setup_gpu_memory_growth():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for g in gpus: tf.config.experimental.set_memory_growth(g, True)
    except Exception: pass

def is_image(path: str) -> bool:
    return os.path.splitext(path.lower())[1] in IMG_EXTS

def list_class_dirs(root_dir: str) -> List[str]:
    return [d for d in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir, d))]

def infer_class_map_from_init(init_root: str) -> Dict[str,int]:
    names = list_class_dirs(init_root)
    if not names: raise ValueError(f"Keine Klassenordner in {init_root} gefunden.")
    return {name: i for i, name in enumerate(sorted(names))}

def scan_labeled_dir(root_dir: str, class_to_idx: Dict[str,int]) -> Tuple[List[str], List[int]]:
    paths, labels = [], []
    for cname, idx in class_to_idx.items():
        cdir = os.path.join(root_dir, cname)
        if not os.path.isdir(cdir): continue
        for p in glob.glob(os.path.join(cdir, "**", "*"), recursive=True):
            if is_image(p):
                paths.append(os.path.abspath(p)); labels.append(idx)
    return paths, labels

def scan_unlabeled_paths(unlabeled_root: str) -> List[str]:
    paths = []
    for p in glob.glob(os.path.join(unlabeled_root, "**", "*"), recursive=True):
        if is_image(p): paths.append(os.path.abspath(p))
    return sorted(paths)

def build_oracle_from_unlabeled(unlabeled_root: str, class_to_idx: Dict[str,int]) -> Dict[str,int]:
    mapping = {}
    for cname, idx in class_to_idx.items():
        cdir = os.path.join(unlabeled_root, cname)
        if not os.path.isdir(cdir): continue
        for p in glob.glob(os.path.join(cdir, "**", "*"), recursive=True):
            if is_image(p): mapping[os.path.abspath(p)] = idx
    return mapping

AUTOTUNE = tf.data.AUTOTUNE
def decode_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    return tf.image.convert_image_dtype(img, tf.float32)

def preprocess(path, label, img_size: int, augment=False):
    img = decode_img(path); img = tf.image.resize(img, (img_size, img_size))
    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.05)
        img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label

def make_ds_from_paths(paths, labels, img_size, batch_size, augment=False, shuffle=False):
    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    if labels is None: label_ds = tf.data.Dataset.from_tensor_slices(tf.fill([len(paths)], -1))
    else: label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int32))
    ds = tf.data.Dataset.zip((path_ds, label_ds))
    if shuffle: ds = ds.shuffle(min(len(paths), 10000), seed=CONFIG["SEED"], reshuffle_each_iteration=True)
    ds = ds.map(lambda p,l: preprocess(p,l,img_size,augment=augment), num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)

def build_mobilenet(num_classes, img_size, weights="imagenet", base_trainable=False):
    base = keras.applications.MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False,
                                          weights=weights if weights in ("imagenet", None) else None)
    base.trainable = base_trainable
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False); x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x); x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs), base

class CosineAnnealer:
    def __init__(self, initial_lr: float, final_lr: float, total_steps: int):
        self.initial_lr = float(initial_lr); self.final_lr = float(final_lr)
        self.total_steps = max(1, int(total_steps))
    def lr_at(self, step: int) -> float:
        s = min(step, self.total_steps)
        cos = 0.5 * (1.0 + math.cos(math.pi * s / self.total_steps))
        return self.final_lr + (self.initial_lr - self.final_lr) * cos

class CosineLRCallback(keras.callbacks.Callback):
    def __init__(self, optimizer: keras.optimizers.Optimizer, annealer: CosineAnnealer, lr_log: list):
        super().__init__(); self.optimizer = optimizer; self.annealer = annealer; self.global_step = 0
        self.lr_log = lr_log  # speichert LR je Batch (global über den Lauf)
    def on_train_batch_begin(self, batch, logs=None):
        lr = self.annealer.lr_at(self.global_step)
        if isinstance(self.optimizer.learning_rate, tf.Variable): self.optimizer.learning_rate.assign(lr)
        else: self.optimizer.learning_rate = lr
        self.global_step += 1
        self.lr_log.append(float(lr))

def build_optimizer_with_cosine(total_steps: int, lr_log: list):
    init_lr = CONFIG["COSINE_INITIAL_LR"]; final_lr = init_lr * CONFIG["COSINE_FINAL_LR_RATIO"]
    lr_var = tf.Variable(init_lr, dtype=tf.float32, trainable=False)
    try: opt = keras.optimizers.experimental.AdamW(learning_rate=lr_var, weight_decay=CONFIG["WEIGHT_DECAY"])
    except Exception: opt = keras.optimizers.Adam(learning_rate=lr_var)
    annealer = CosineAnnealer(init_lr, final_lr, total_steps); cosine_cb = CosineLRCallback(opt, annealer, lr_log)
    return opt, cosine_cb

def compile_with_optimizer(model, optimizer):
    model.compile(optimizer=optimizer, loss=keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    return model

def predict_probs(model, ds): return model.predict(ds, verbose=0)
def entropy_scores(p): 
    eps=1e-12; return -np.sum(p * np.log(np.clip(p, eps, 1.0)), axis=1)
def least_confidence_scores(p): return np.max(p, axis=1)
def margin_scores(p): 
    top2=np.sort(p, axis=1)[:, -2:]; return top2[:,1]-top2[:,0]

def select_uncertain_indices(p, k, strategy):
    if strategy=="entropy": order=np.argsort(-entropy_scores(p))
    elif strategy=="least_confidence": order=np.argsort(least_confidence_scores(p))
    elif strategy=="margin": order=np.argsort(margin_scores(p))
    else: raise ValueError("Unbekannte Strategie")
    return order[:k]

def select_high_confidence_indices(p, delta, normalize, num_classes, min_conf, min_margin):
    H_raw = entropy_scores(p); H_norm = H_raw / math.log(num_classes)
    maxprob = np.max(p, axis=1); top2 = np.sort(p, axis=1)[:, -2:]; margin = top2[:,1]-top2[:,0]
    mask = (H_norm < delta) if normalize else (H_raw < delta)
    mask = mask & (maxprob >= float(min_conf)) & (margin >= float(min_margin))
    idx = np.where(mask)[0]; pseudo = np.argmax(p[idx], axis=1)
    return idx, pseudo, H_raw[idx], H_norm[idx], maxprob[idx], margin[idx]

def current_delta(iter_idx): return max(0.0, CONFIG["DELTA0"] - iter_idx * CONFIG["DECAY"])

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def save_line_plot(x, y, title, xlabel, ylabel, outpath):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def save_dual_line_plot(x, y1, y2, label1, label2, title, xlabel, ylabel, outpath):
    plt.figure()
    plt.plot(x, y1, label=label1)
    plt.plot(x, y2, label=label2)
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def main():
    print("TensorFlow:", tf.__version__)
    seed_all(CONFIG["SEED"]); setup_gpu_memory_growth()
    ROOT = CONFIG["DATA_ROOT"]
    init_root = os.path.join(ROOT, CONFIG["INIT_LABELED_DIR"])
    unl_root  = os.path.join(ROOT, CONFIG["UNLABELED_DIR"])
    val_root  = os.path.join(ROOT, CONFIG["VAL_DIR"])
    test_root = os.path.join(ROOT, CONFIG["TEST_DIR"])

    plots_dir = ensure_dir(os.path.join(ROOT, CONFIG["PLOTS_DIR"]))
    csv_dir   = ensure_dir(os.path.join(ROOT, CONFIG["CSV_DIR"]))

    class_to_idx = infer_class_map_from_init(init_root); idx_to_class = {v:k for k,v in class_to_idx.items()}
    if CONFIG["EXPECT_NUM_CLASSES"] and len(class_to_idx)!=CONFIG["EXPECT_NUM_CLASSES"]:
        print(f"[Warn] Erwartet {CONFIG['EXPECT_NUM_CLASSES']} Klassen, gefunden: {len(class_to_idx)}")
    NUM_CLASSES = len(class_to_idx); print("Klassen:", class_to_idx)

    init_paths, init_labels = scan_labeled_dir(init_root, class_to_idx)
    val_paths,  val_labels  = scan_labeled_dir(val_root,  class_to_idx)
    test_paths, test_labels = scan_labeled_dir(test_root, class_to_idx)
    unlabeled_paths = scan_unlabeled_paths(unl_root)
    oracle_map = build_oracle_from_unlabeled(unl_root, class_to_idx)

    val_ds  = make_ds_from_paths(val_paths,  val_labels,  CONFIG["IMG_SIZE"], CONFIG["BATCH_SIZE"])
    test_ds = make_ds_from_paths(test_paths, test_labels, CONFIG["IMG_SIZE"], CONFIG["BATCH_SIZE"])

    labeled_paths = list(init_paths); labeled_labels = list(init_labels)

    # Cosine steps estimate
    steps_warmup = max(1, math.ceil(len(labeled_paths)/CONFIG["BATCH_SIZE"])) * CONFIG["WARMUP_EPOCHS"]
    steps_per_epoch_upper = max(1, math.ceil((len(labeled_paths)+len(unlabeled_paths))/CONFIG["BATCH_SIZE"]))
    total_steps_est = steps_warmup + CONFIG["ITERS"]*CONFIG["EPOCHS_PER_ITER"]*steps_per_epoch_upper
    print(f"[Cosine] Gesamt-Trainingsschritte (grobe Schätzung): {total_steps_est}")

    # LR logging (global per batch)
    lr_log = []

    model, base = build_mobilenet(NUM_CLASSES, CONFIG["IMG_SIZE"], weights=CONFIG["WEIGHTS"], base_trainable=False)
    if CONFIG["USE_COSINE"]:
        optimizer, cosine_cb = build_optimizer_with_cosine(total_steps_est, lr_log)
    else:
        lr_var = tf.Variable(CONFIG["COSINE_INITIAL_LR"], dtype=tf.float32, trainable=False)
        try: optimizer = keras.optimizers.experimental.AdamW(learning_rate=lr_var, weight_decay=CONFIG["WEIGHT_DECAY"])
        except Exception: optimizer = keras.optimizers.Adam(learning_rate=lr_var)
        cosine_cb = None
    model = compile_with_optimizer(model, optimizer)
    callbacks = [cb for cb in [cosine_cb] if cb is not None]

    # Training-History über alle Epochen (Warmup + Iterationen)
    global_epoch = 0
    train_hist = {"epoch": [], "loss": [], "acc": [], "val_loss": [], "val_acc": []}

    def log_history(hist):
        nonlocal global_epoch, train_hist
        losses = hist.history.get("loss", [])
        accs   = hist.history.get("accuracy", [])
        vloss  = hist.history.get("val_loss", [])
        vacc   = hist.history.get("val_accuracy", [])
        for i in range(len(losses)):
            global_epoch += 1
            train_hist["epoch"].append(global_epoch)
            train_hist["loss"].append(float(losses[i]))
            train_hist["acc"].append(float(accs[i]) if i < len(accs) else None)
            train_hist["val_loss"].append(float(vloss[i]) if i < len(vloss) else None)
            train_hist["val_acc"].append(float(vacc[i]) if i < len(vacc) else None)

    # Warmup
    warmup_ds = make_ds_from_paths(labeled_paths, labeled_labels, CONFIG["IMG_SIZE"], CONFIG["BATCH_SIZE"], augment=True, shuffle=True)
    print("Warmup-Training startet...")
    h = model.fit(warmup_ds, epochs=CONFIG["WARMUP_EPOCHS"], validation_data=val_ds, verbose=1, callbacks=callbacks)
    log_history(h)

    all_stats = []
    for it in range(1, CONFIG["ITERS"]+1):
        print(f"\n=== Iteration {it}/{CONFIG['ITERS']} ===")
        if it == CONFIG["BASE_TRAINABLE_AFTER"]:
            print("[Info] Unfreezing MobileNetV2 backbone.")
            base.trainable = True
            model = compile_with_optimizer(model, optimizer)  # gleicher Optimizer

        if len(unlabeled_paths)==0: print("[Info] Keine unlabeled Bilder mehr – Abbruch."); break

        unl_ds = make_ds_from_paths(unlabeled_paths, None, CONFIG["IMG_SIZE"], CONFIG["BATCH_SIZE"])
        probs = predict_probs(model, unl_ds)

        K = min(CONFIG["K"], len(unlabeled_paths))
        uncertain_local = select_uncertain_indices(probs, K, CONFIG["STRATEGY"]) if K>0 else np.array([],dtype=int)
        uncertain_global = [unlabeled_paths[i] for i in uncertain_local]

        delta = current_delta(it)
        hc_local, hc_pseudo, H_raw_sub, H_norm_sub, maxprob_sub, margin_sub = select_high_confidence_indices(
            probs, delta, CONFIG["NORMALIZE_ENTROPY"], NUM_CLASSES,
            CONFIG["PSEUDO_MIN_CONF"], CONFIG["PSEUDO_MIN_MARGIN"]
        )
        uncertain_set = set(uncertain_local.tolist())
        keep_mask = ~np.isin(hc_local, list(uncertain_set))
        hc_local = hc_local[keep_mask]; hc_pseudo = hc_pseudo[keep_mask]
        H_raw_sub = H_raw_sub[keep_mask]; H_norm_sub = H_norm_sub[keep_mask]
        maxprob_sub = maxprob_sub[keep_mask]; margin_sub = margin_sub[keep_mask]
        hc_paths = [unlabeled_paths[i] for i in hc_local]

        newly_labeled = 0
        for pth in uncertain_global:
            ap = os.path.abspath(pth); t = oracle_map.get(ap, None)
            if t is not None:
                labeled_paths.append(pth); labeled_labels.append(t); newly_labeled += 1
        unlabeled_paths = [p for i,p in enumerate(unlabeled_paths) if i not in uncertain_set]
        print(f"[Info] Annotiert (Ordner-Oracle): {newly_labeled}/{len(uncertain_global)}; unlabeled verbleibend: {len(unlabeled_paths)}")

        # CSV + Accuracy
        csv_path = os.path.join(csv_dir, f"pseudolabels_iter_{it:02d}.csv")
        header = ["iteration","path","true_idx","true_name","pseudo_idx","pseudo_name","correct","entropy","entropy_norm","max_prob","margin"]
        correct_count = 0; rows=[]
        for j, pth in enumerate(hc_paths):
            ap = os.path.abspath(pth); t = oracle_map.get(ap, None); ps = int(hc_pseudo[j])
            H_raw = float(H_raw_sub[j]); H_norm = float(H_norm_sub[j]); maxp = float(maxprob_sub[j]); marg = float(margin_sub[j])
            if t is not None:
                ok = (ps==t); correct_count += int(ok); ok_str = "true" if ok else "false"
                rows.append([it, ap, t, idx_to_class[t], ps, idx_to_class[ps], ok_str, H_raw, H_norm, maxp, marg])
            else:
                rows.append([it, ap, -1, "", ps, idx_to_class[ps], "n/a", H_raw, H_norm, maxp, marg])
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(rows)
        print(f"[CSV] Pseudolabels gespeichert: {csv_path}")

        num_with_true = sum(1 for r in rows if isinstance(r[2], int) and r[2] >= 0)
        pseudo_acc = (correct_count / num_with_true) if num_with_true>0 else None
        print(f"[Info] Pseudolabels: {len(hc_paths)} | mit True-Label: {num_with_true} | Accuracy: {pseudo_acc if pseudo_acc is not None else 'n/a'}")

        # Train
        train_true = make_ds_from_paths(labeled_paths, labeled_labels, CONFIG["IMG_SIZE"], CONFIG["BATCH_SIZE"], augment=True, shuffle=True)
        if len(hc_paths)>0:
            train_hc = make_ds_from_paths(hc_paths, list(hc_pseudo), CONFIG["IMG_SIZE"], CONFIG["BATCH_SIZE"], augment=True, shuffle=True)
            train_ds = train_true.concatenate(train_hc)
        else: train_ds = train_true

        h = model.fit(train_ds, epochs=CONFIG["EPOCHS_PER_ITER"], validation_data=val_ds, verbose=1, callbacks=callbacks)
        log_history(h)

        test_loss, test_acc = model.evaluate(test_ds, verbose=0)
        all_stats.append({
            "iter": it, "delta": float(delta),
            "labeled": len(labeled_paths), "unlabeled": len(unlabeled_paths),
            "high_conf": len(hc_paths), "pseudo_acc": (float(pseudo_acc) if pseudo_acc is not None else None),
            "test_loss": float(test_loss), "test_acc": float(test_acc),
        })
        print(f"[Iter {it}] δ={delta:.3f} | labeled={len(labeled_paths)} | unlabeled={len(unlabeled_paths)} | HC={len(hc_paths)} | pseudo_acc={all_stats[-1]['pseudo_acc']} | test_acc={test_acc:.4f}")

    OUT_STATS = os.path.join(ROOT, "ceal_stats.json")
    with open(OUT_STATS, "w", encoding="utf-8") as f: json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print("Stats gespeichert nach:", OUT_STATS)
    print("Pseudolabel-CSV-Dateien liegen in:", csv_dir)

    # ======================
    # Plots am Ende erzeugen
    # ======================
    # 1) Test-Accuracy über Iterationen
    its = [d["iter"] for d in all_stats]
    test_acc = [d["test_acc"] for d in all_stats]
    save_line_plot(its, test_acc, "Test Accuracy über Iterationen", "Iteration", "Test Accuracy",
                   os.path.join(plots_dir, "test_accuracy_over_iters.png"))

    # 2) Pseudolabel-Accuracy
    pseudo_acc = [d["pseudo_acc"] if d["pseudo_acc"] is not None else np.nan for d in all_stats]
    save_line_plot(its, pseudo_acc, "Pseudolabel Accuracy über Iterationen", "Iteration", "Pseudo-Acc",
                   os.path.join(plots_dir, "pseudo_accuracy_over_iters.png"))

    # 3) Labeled/Unlabeled-Verlauf
    labeled_counts   = [d["labeled"]   for d in all_stats]
    unlabeled_counts = [d["unlabeled"] for d in all_stats]
    save_dual_line_plot(its, labeled_counts, unlabeled_counts, "labeled", "unlabeled",
                        "Labeled/Unlabeled über Iterationen", "Iteration", "Anzahl",
                        os.path.join(plots_dir, "labeled_unlabeled_over_iters.png"))

    # 4) High-Confidence pro Iteration
    hc_counts = [d["high_conf"] for d in all_stats]
    save_line_plot(its, hc_counts, "High-Confidence Pseudolabels pro Iteration", "Iteration", "Anzahl HC",
                   os.path.join(plots_dir, "high_conf_over_iters.png"))

    # 5) Training über Epochen (Loss/Accuracy)
    save_dual_line_plot(train_hist["epoch"], train_hist["loss"], train_hist["val_loss"],
                        "train_loss", "val_loss", "Loss über Epochen", "Epoche", "Loss",
                        os.path.join(plots_dir, "loss_over_epochs.png"))
    save_dual_line_plot(train_hist["epoch"], train_hist["acc"], train_hist["val_acc"],
                        "train_acc", "val_acc", "Accuracy über Epochen", "Epoche", "Accuracy",
                        os.path.join(plots_dir, "accuracy_over_epochs.png"))

    # 6) LR-Schedule über Batches (global, inkl. Warmup & alle Iterationen)
    if len(lr_log) > 0:
        save_line_plot(list(range(1, len(lr_log)+1)), lr_log,
                       "Learning Rate (Cosine) über Batches", "Batch (global)", "LR",
                       os.path.join(plots_dir, "lr_cosine_over_batches.png"))

    print("Plots gespeichert in:", plots_dir)

if __name__ == "__main__":
    main()
