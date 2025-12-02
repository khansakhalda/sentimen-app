"""
pipeline.py
Pipeline satu-berkas: Collect → Clean → Casefold → Normalize → Tokenize → Label → TF-IDF
→ Split → SVM Linear → SVM RBF
"""

import os, sys, csv, time, html, subprocess
from datetime import datetime, timezone
from urllib.parse import urlparse, parse_qs
import pandas as pd
from dateutil import parser as dtparser
from tqdm import tqdm
from joblib import parallel_backend
from helpers import load_slang_map, normalize_with_slang

DATA_DIR = "data"
DEFAULT_FROM_DATE = "2024-09-01"
DEFAULT_TO_DATE   = "2025-08-31"
BASE_SIMPLE = "app_source_getcontact"

TRAIN_FILE = None
TEST_FILE  = None

LAST_TUNING_RESULTS_LINEAR = None
LAST_TUNING_RESULTS_RBF = None

def ensure_dir(d=DATA_DIR): os.makedirs(d, exist_ok=True)
def now_tag(): return datetime.now().strftime("%Y%m%d-%H%M%S")

def show_download_button(path: str, label: str):
    if "google.colab" not in sys.modules: return
    try:
        from google.colab import output, files
        from IPython.display import HTML, display
        key = "dl_" + os.path.basename(path).replace(".", "_")
        def _dl(_=None): files.download(path)
        output.register_callback(f'notebook.{key}', _dl)
        display(HTML(f"""
        <button onclick="google.colab.kernel.invokeFunction('notebook.{key}', [], {{}})"
                style="background:#0d6efd;color:white;border:none;padding:8px 14px;border-radius:6px;cursor:pointer;margin:6px 0;">
          ⬇️ {label}
        </button>
        """))
    except Exception:
        pass

def latest_required(prefix: str, err: str):
    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)
             if f.startswith(prefix) and f.endswith(".csv")]
    if not files:
        raise FileNotFoundError(err)
    files.sort()
    return files[-1]

def safe_import_gps():
    try:
        from google_play_scraper import reviews, Sort, app, search
        return reviews, Sort, app, search
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "google-play-scraper>=1.2.6"])
        from google_play_scraper import reviews, Sort, app, search
        return reviews, Sort, app, search

def safe_import_viz():
    try:
        import matplotlib.pyplot as plt
        from wordcloud import WordCloud
        return plt, WordCloud
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "matplotlib", "wordcloud"])
        import matplotlib.pyplot as plt
        from wordcloud import WordCloud
        return plt, WordCloud

def safe_import_nltk():
    try:
        import nltk
        from nltk.tokenize import word_tokenize
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "nltk"])
        import nltk
        from nltk.tokenize import word_tokenize
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    return nltk, word_tokenize

def _to_utc(dt):
    if dt is None: return None
    if getattr(dt, "tzinfo", None) is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def _extract_id_from_url(s: str):
    try:
        u = urlparse(s)
        if "play.google.com" in u.netloc and "id=" in s:
            q = parse_qs(u.query)
            if "id" in q and q["id"]: return q["id"][0]
            return s.split("id=")[-1].split("&")[0]
    except Exception:
        pass
    return None

def resolve_app_id(app_input: str, lang="id", country="id"):
    reviews_fn, Sort, app_fn, search_fn = safe_import_gps()
    pkg = _extract_id_from_url(app_input.strip())
    if pkg: return pkg
    try:
        detail = app_fn(app_input, lang=lang, country=country)
        if detail.get("appId"): return detail["appId"]
    except Exception:
        pass
    hits = search_fn(app_input, lang=lang, country=country, n=10)
    if not hits: raise ValueError("Gagal resolve app id.")
    tokens = [t for t in app_input.lower().split() if t.strip()]
    for h in hits:
        title = (h.get("title") or "").lower()
        if all(t in title for t in tokens): return h["appId"]
    return hits[0]["appId"]

def _build_word_freq(texts):
    import re
    from collections import Counter
    tokens = []
    for t in texts:
        tokens += re.findall(r"\b\w+\b", str(t).lower())
    return Counter(tokens)

# def _preproc_visuals(csv_path: str, out_dir: str, base_prefix: str):
#     plt, WordCloud = safe_import_viz()
#     df_raw = pd.read_csv(csv_path, encoding="utf-8-sig")

#     if "content" not in df_raw.columns:
#         return None, None

#     # Ambil semua teks
#     texts = df_raw["content"].fillna("").astype(str).tolist()
#     joined = "\n".join(texts)

#     # Hitung frekuensi kata
#     freqs = _build_word_freq(texts)

#     # ================== WORDCLOUD ==================
#     wc = WordCloud(width=1000, height=600, background_color="white").generate(joined)
#     wc_path = os.path.join(out_dir, f"wc_{base_prefix}.png")

#     fig1 = plt.figure(figsize=(8, 5))
#     plt.imshow(wc)
#     plt.axis("off")
#     plt.tight_layout()
#     fig1.savefig(wc_path, dpi=120, bbox_inches="tight")
#     plt.close(fig1)

#     # ================== FREKUENSI BAR CHART ==================
#     topN = 15
#     top = freqs.most_common(topN)

#     if not top:
#         return wc_path, None

#     words, counts = zip(*top)

#     freq_path = os.path.join(out_dir, f"freq_{base_prefix}.png")

#     fig2, ax2 = plt.subplots(figsize=(9, 5))

#     # posisi dibalik agar yang frekuensinya besar muncul paling atas
#     y_pos = list(range(len(words)))[::-1]
#     vals = list(counts)[::-1]

#     ax2.barh(y_pos, vals)
#     ax2.set_yticks(y_pos)
#     ax2.set_yticklabels(words[::-1])
#     ax2.set_xlabel("Frekuensi")
#     ax2.set_title("Top-15 Frekuensi Kata (sebelum preprocessing)")

#     # ===== FORMAT ANGKA INDONESIA =====
#     def format_indo(n: int) -> str:
#         return f"{n:,}".replace(",", ".")

#     offset = max(vals) * 0.02 

#     for y, v in zip(y_pos, vals):
#         ax2.text(v + offset, y, format_indo(v), va="center", fontsize=10)

#     fig2.tight_layout()
#     fig2.savefig(freq_path, dpi=120, bbox_inches="tight")
#     plt.close(fig2)

#     return wc_path, freq_path

def stage1_collect(app_input: str, out_dir=DATA_DIR, max_reviews=0, page_size=200,
                   lang="id", country="id", sleep_sec=0.0,
                   from_date=DEFAULT_FROM_DATE, to_date=DEFAULT_TO_DATE,
                   progress_callback=None):
    ensure_dir(out_dir)
    reviews_fn, Sort, app_fn, search_fn = safe_import_gps()
    start_dt = _to_utc(dtparser.parse(from_date))
    end_dt   = _to_utc(dtparser.parse(to_date)) + pd.Timedelta(days=1)

    pkg = resolve_app_id(app_input, lang=lang, country=country)
    safe_pkg = pkg.replace(".", "_")
    out_path = os.path.join(out_dir, f"raw_{safe_pkg}.csv")

    rows, seen_ids = [], set()
    token = None; total_fetched = 0
    seen_tokens = set()
    print(f"[Collect] {pkg} | {from_date}..{to_date}")
    bar = tqdm(desc="Mengunduh ulasan", unit="ulasan", total=None if max_reviews==0 else max_reviews)

    try:
        while True:
            if max_reviews and total_fetched >= max_reviews: break
            count = page_size if max_reviews==0 else min(page_size, max_reviews-total_fetched)
            batch, token = reviews_fn(pkg, lang=lang, country=country, sort=Sort.NEWEST, count=count, continuation_token=token)
            if not batch: break

            if token in seen_tokens:
                print("[Collect] Detected repeated continuation_token. Stopping."); break
            seen_tokens.add(token)

            at_list = []
            for r in batch:
                at_dt = _to_utc(r.get("at"))
                if at_dt is None: continue
                at_list.append(at_dt)

                if not (start_dt <= at_dt < end_dt): continue
                rid = r.get("reviewId")
                if rid in seen_ids: continue
                seen_ids.add(rid)

                def clean(x):
                    x = "" if not x else str(x)
                    return html.unescape(" ".join(x.split()))
                rows.append({
                    "reviewId": rid,
                    "userName": clean(r.get("userName")),
                    "content": clean(r.get("content")),
                    "score": r.get("score"),
                    "at": at_dt.strftime("%Y-%m-%d %H:%M:%S")
                })
                total_fetched += 1
                bar.update(1)
                if progress_callback: progress_callback(total_fetched)
                if max_reviews and total_fetched >= max_reviews: break

            if at_list and min(at_list) < start_dt:
                print("[Collect] Reached older than from_date. Stopping."); break
            if token is None: break
            if sleep_sec > 0: time.sleep(sleep_sec)
    finally:
        bar.close()

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
    print(f"✅ Simpan: {len(df)} ulasan → {out_path}")

    # wc_raw, freq_raw = _preproc_visuals(out_path, out_dir, safe_pkg)
    # return out_path, os.path.abspath(wc_raw), os.path.abspath(freq_raw)

# ---------- Cleaning / Casefold / Normalize / Tokenize ----------
def stage2_clean():
    raw = sorted([f for f in os.listdir(DATA_DIR) if f.startswith("raw_") and f.endswith(".csv")])
    if not raw: raise FileNotFoundError("Tidak ada file raw_*.csv")
    path = os.path.join(DATA_DIR, raw[-1])
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["content_clean_nocase"] = df["content"].fillna("").astype(str).str.replace(r"[^A-Za-z\s]", " ", regex=True)
    out = os.path.join(DATA_DIR, f"clean_no_case_{BASE_SIMPLE}.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"✅ Cleaning → {out}")
    return out

def stage3_casefold():
    path = os.path.join(DATA_DIR, f"clean_no_case_{BASE_SIMPLE}.csv")
    if not os.path.exists(path): raise FileNotFoundError("Jalankan tahap 2 dulu.")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["content_clean"] = df["content_clean_nocase"].astype(str).str.lower()
    out = os.path.join(DATA_DIR, f"clean_casefold_{BASE_SIMPLE}.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"✅ Casefold → {out}")
    return out

def stage4_normalize():
    path = os.path.join(DATA_DIR, f"clean_casefold_{BASE_SIMPLE}.csv")
    if not os.path.exists(path): raise FileNotFoundError("Jalankan tahap 3 dulu.")
    df = pd.read_csv(path, encoding="utf-8-sig")

    slang_map = {}
    if os.path.exists("slangs.txt"):
        for line in open("slangs.txt", "r", encoding="utf-8"):
            line = line.strip()
            if not line or ":" not in line: continue
            k, v = line.split(":", 1)
            slang_map[k.strip().lower()] = v.strip().lower()

    slang_map = load_slang_map("slangs.txt")

    df["content_norm"] = df["content_clean"].astype(str).map(
        lambda s: normalize_with_slang(s, slang_map)
    )

    before = len(df)
    df = df[df["content_norm"].astype(str).str.strip() != ""]
    after = len(df)
    print(f"✅ Normalisasi → {after:,} ulasan tersisa (hapus {before - after:,} kosong)")

    out = os.path.join(DATA_DIR, f"norm_{BASE_SIMPLE}.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return out

def stage5_tokenize():
    nltk, word_tokenize = safe_import_nltk()
    path = os.path.join(DATA_DIR, f"norm_{BASE_SIMPLE}.csv")
    if not os.path.exists(path): raise FileNotFoundError("Jalankan tahap 4 dulu.")
    df = pd.read_csv(path, encoding="utf-8-sig")

    df["tokens"] = df["content_norm"].astype(str).map(word_tokenize)
    df["tokens_join"] = df["tokens"].map(lambda xs: " ".join(xs))

    before = len(df)
    df = df[df["tokens_join"].astype(str).str.strip() != ""]
    after = len(df)
    print(f"✅ Tokenizing → {after:,} ulasan tersisa (hapus {before - after:,} kosong)")

    out = os.path.join(DATA_DIR, f"token_{BASE_SIMPLE}.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return out

# ---------- InSet Labelling ----------
def _load_inset(pos="positive.tsv", neg="negative.tsv"):
    def _read(p):
        if not os.path.exists(p): return pd.DataFrame(columns=["word","weight"])
        df = pd.read_csv(p, sep="\t", header=None, names=["word","weight"],
                         encoding="utf-8-sig", dtype=str, engine="python")
        df["word"] = df["word"].astype(str).str.strip().str.lower()
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
        return df.dropna()
    posdf = _read(pos); negdf = _read(neg)
    if posdf.empty and negdf.empty:
        raise FileNotFoundError("positive.tsv & negative.tsv tidak ditemukan.")
    lex = {}
    for _,r in posdf.iterrows(): lex[r["word"]] = lex.get(r["word"],0.0)+float(r["weight"])
    for _,r in negdf.iterrows(): lex[r["word"]] = lex.get(r["word"],0.0)+float(r["weight"])
    return lex

def stage6_label_inset():
    token_path = os.path.join(DATA_DIR, f"token.csv")
    if not os.path.exists(token_path):
        raise FileNotFoundError("Jalankan tahap 5 dulu (tokenize).")
    df = pd.read_csv(token_path, encoding="utf-8-sig")
    if "tokens_join" not in df.columns: raise KeyError("Kolom tokens_join hilang.")
    lex = _load_inset()

    df["sent_score_raw"] = df["tokens_join"].fillna("").astype(str).apply(
        lambda s: sum(lex.get(t,0.0) for t in s.split())
    )
    df["sent_score"] = df["sent_score_raw"].map(lambda x: max(-5.0, min(5.0, x)))
    df["label"] = df["sent_score"].map(lambda s: (1 if s > 0 else (-1 if s < 0 else 0)))

    out = os.path.join(DATA_DIR, f"label_only_{BASE_SIMPLE}.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL, lineterminator="\n")
    print(f"✅ Labelling InSet → {out} (total dokumen: {len(df):,})")
    return out

# ---------- TF-IDF ----------
def stage7_tfidf():
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    label_path = os.path.join(DATA_DIR, f"label_only_{BASE_SIMPLE}.csv")
    if not os.path.exists(label_path):
        raise FileNotFoundError("Jalankan tahap 6 dulu.")
    df = pd.read_csv(label_path, encoding="utf-8-sig")
    if "tokens_join" not in df.columns:
        raise KeyError("Kolom tokens_join hilang.")

    TOKEN_PATTERN = r"(?u)\b[a-z]{4,}\b"
    MIN_DF = 3
    MAX_DF = 0.98

    vec = TfidfVectorizer(
        analyzer="word",
        token_pattern=TOKEN_PATTERN,
        lowercase=False,
        ngram_range=(1,1),
        min_df=MIN_DF,
        max_df=MAX_DF,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False
    )

    # ===== Hitung matriks TF-IDF =====
    X = vec.fit_transform(df["tokens_join"].fillna("").astype(str))

    vocab = vec.get_feature_names_out()
    idf = vec.idf_

    # ==============================================
    # 🔥 Tambahan penting: hitung TF-IDF maksimum per term
    # ==============================================
    tfidf_max = np.max(X.toarray(), axis=0)
    # ===============================================

    # Simpan file fitur TF-IDF yang sekarang lengkap dengan kolom tfidf
    out_feat = os.path.join(DATA_DIR, f"tfidf_features_{BASE_SIMPLE}.csv")
    pd.DataFrame({
        "term": vocab,
        "idf": idf,
        "tfidf": tfidf_max
    }).sort_values("idf", ascending=False).to_csv(
        out_feat,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL,
        lineterminator="\n"
    )

    # ===== Simpan top-5 TF-IDF tiap dokumen seperti sebelumnya =====
    def topk_row(row, k=5):
        if row.nnz == 0:
            return ""
        inds, vals = row.indices, row.data
        order = vals.argsort()[::-1][:k]
        return ", ".join([f"{vocab[inds[i]]}:{vals[i]:.3f}" for i in order])

    topk = [topk_row(X.getrow(i), k=5) for i in range(X.shape[0])]

    out_top = os.path.join(DATA_DIR, f"tfidf_doc_top5_{BASE_SIMPLE}.csv")
    pd.DataFrame({
        "reviewId": df["reviewId"] if "reviewId" in df.columns else range(len(topk)),
        "label": df["label"] if "label" in df.columns else None,
        "top5_tfidf": topk
    }).to_csv(
        out_top,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL,
        lineterminator="\n"
    )

    print(f"✅ TF-IDF → {out_feat} | {out_top}")
    return out_feat

# ---------- Split ----------
def stage8_split(frac, random_seed=42):
    """
    Melakukan split.
    Digunakan UI untuk menampilkan progress per-rasio.
    """

    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer

    label_path = os.path.join(DATA_DIR, f"label_only_{BASE_SIMPLE}.csv")
    if not os.path.exists(label_path):
        raise FileNotFoundError("Jalankan tahap 6 dulu (labelling).")

    df = pd.read_csv(label_path, encoding="utf-8-sig")

    TOKEN_PATTERN = r"(?u)\b[a-z]{4,}\b"
    vec = TfidfVectorizer(
        analyzer="word",
        token_pattern=TOKEN_PATTERN,
        lowercase=False,
        ngram_range=(1, 1),
        min_df=3,
        max_df=0.98,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False
    )

    X_all = vec.fit_transform(df["tokens_join"].fillna("").astype(str)).toarray()
    y_all = df["label"].astype(str).values
    vocab = vec.get_feature_names_out()

    # split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all,
        test_size=1 - frac,
        random_state=random_seed,
        stratify=y_all,
        shuffle=True
    )

    # dataframe
    df_train = pd.DataFrame(X_tr, columns=vocab)
    df_test = pd.DataFrame(X_te, columns=vocab)
    df_train["label"] = y_tr
    df_test["label"] = y_te

    # generate key contoh: 0.8 → 80_20
    key = f"{int(frac*100)}_{int((1-frac)*100)}"

    train_path = os.path.join(DATA_DIR, f"train_{key}_{BASE_SIMPLE}.csv")
    test_path = os.path.join(DATA_DIR, f"test_{key}_{BASE_SIMPLE}.csv")

    df_train.to_csv(train_path, index=False, encoding="utf-8-sig")
    df_test.to_csv(test_path, index=False, encoding="utf-8-sig")

    return train_path, test_path, len(df_train), len(df_test)

# ---------- SVM Linear ----------
def stage9_svm_linear():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import confusion_matrix, classification_report

    global TRAIN_FILE, TEST_FILE

    train_df = pd.read_csv(TRAIN_FILE, encoding="utf-8-sig")
    test_df  = pd.read_csv(TEST_FILE,  encoding="utf-8-sig")

    LABEL_COL = "label"
    Xtr = train_df.drop(columns=[LABEL_COL]).values
    ytr = train_df[LABEL_COL].astype(str).values
    Xte = test_df.drop(columns=[LABEL_COL]).values
    yte = test_df[LABEL_COL].astype(str).values

    # ----- GridSearchCV -----
    cv_used = 3
    grid = GridSearchCV(
        LinearSVC(random_state=42, max_iter=6000),
        param_grid={"C": [0.1, 1, 10, 100]},
        cv=cv_used,
        scoring="accuracy",
        return_train_score=True
    )
    grid.fit(Xtr, ytr)

    pred = grid.best_estimator_.predict(Xte)

    best_params = grid.best_params_
    best_score = grid.best_score_
    cv_results_df = pd.DataFrame(grid.cv_results_)

    # ----- Confusion Matrix -----
    labels = sorted(list(set(ytr) | set(yte)))
    cm = confusion_matrix(yte, pred, labels=labels)

    cm_df = pd.DataFrame(
        cm,
        columns=[f"Pred_{l}" for l in labels],
        index=[f"Actual_{l}" for l in labels]
    )

    report_text = (
        f"Model: LinearSVC\n"
        f"K-Fold CV: {cv_used}\n"
        f"Best Params: {best_params}\n\n"
        + classification_report(yte, pred, labels=labels, digits=4)
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    plt.close(fig)

    # return 7 item
    return (
        report_text,
        fig,
        cm_df,
        best_params,
        best_score,
        cv_results_df,
        cv_used
    )

# ---------- SVM RBF ----------
def stage10_svm_rbf(progress_callback=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import confusion_matrix, classification_report

    global TRAIN_FILE, TEST_FILE

    train_df = pd.read_csv(TRAIN_FILE, encoding="utf-8-sig")
    test_df  = pd.read_csv(TEST_FILE,  encoding="utf-8-sig")

    LABEL_COL = "label"
    X = train_df.drop(columns=[LABEL_COL]).values
    y = train_df[LABEL_COL].astype(str).values

    chunks = np.array_split(np.arange(len(X)), 5)
    candidate_params = []

    total_chunks = len(chunks)

    # ---- Tuning manual setiap chunk ----
    for idx, chunk in enumerate(chunks):

        if progress_callback:
            progress_callback(f"🔵 Chunk {idx+1}/{total_chunks} — mulai tuning...")

        X_sub = X[chunk]
        y_sub = y[chunk]

        param_grid = {
            "C":     [0.1, 1, 10, 100],
            "gamma": [1, 0.1, 0.01, 0.001],
        }

        grid_list = [(C, g) for C in param_grid["C"] for g in param_grid["gamma"]]
        n_total = len(grid_list)

        best_score = -1
        best_param = None

        from sklearn.model_selection import cross_val_score

        for i, (C_val, g_val) in enumerate(grid_list, start=1):

            if progress_callback:
                progress_callback(
                    f"🔵 Chunk {idx+1}/{total_chunks} — Grid {i}/{n_total} (C={C_val}, gamma={g_val})"
                )

            model = SVC(kernel="rbf", C=C_val, gamma=g_val)

            scores = cross_val_score(model, X_sub, y_sub, cv=3, scoring="accuracy")
            mean_acc = scores.mean()

            if mean_acc > best_score:
                best_score = mean_acc
                best_param = {"C": C_val, "gamma": g_val}

        if progress_callback:
            progress_callback(
                f"✔ Chunk {idx+1}/{total_chunks} — Best params: {best_param}"
            )

        candidate_params.append(best_param)

    # ---- Parameter final ----
    df_params = pd.DataFrame(candidate_params)
    C_best = df_params["C"].mode()[0]
    gamma_best = df_params["gamma"].mode()[0]

    if progress_callback:
        progress_callback(f"🏁 PARAMETER FINAL: C={C_best}, gamma={gamma_best}")

    # ---- Train akhir full data ----
    final_model = SVC(kernel="rbf", C=C_best, gamma=gamma_best)

    if progress_callback:
        progress_callback("⏳ Melatih model akhir dengan seluruh data...")

    final_model.fit(X, y)

    if progress_callback:
        progress_callback("✔ Training model akhir selesai!")

    # ---- Evaluasi ----
    Xte = test_df.drop(columns=[LABEL_COL]).values
    yte = test_df[LABEL_COL].astype(str).values
    pred = final_model.predict(Xte)

    labels = sorted(list(set(y) | set(yte)))
    cm = confusion_matrix(yte, pred, labels=labels)
    cm_df = pd.DataFrame(cm, columns=[f"Pred_{l}" for l in labels],
                               index=[f"Actual_{l}" for l in labels])

    report_text = classification_report(yte, pred, labels=labels, digits=4)

    fig, ax = plt.subplots(figsize=(5,4))
    ax.imshow(cm, cmap="Blues")
    plt.close(fig)

    return (
        report_text,
        fig,
        cm_df,
        {"C": C_best, "gamma": gamma_best},
        None,
        df_params,
        3
    )

# def stage10_svm_rbf(subsample_frac=1.0):
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from sklearn.pipeline import make_pipeline
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.kernel_approximation import RBFSampler
#     from sklearn.linear_model import SGDClassifier
#     from sklearn.model_selection import GridSearchCV
#     from sklearn.metrics import confusion_matrix, classification_report

#     global TRAIN_FILE, TEST_FILE

#     train_df = pd.read_csv(TRAIN_FILE, encoding="utf-8-sig")
#     test_df  = pd.read_csv(TEST_FILE,  encoding="utf-8-sig")

#     LABEL_COL = "label"
#     Xtr = train_df.drop(columns=[LABEL_COL]).values
#     ytr = train_df[LABEL_COL].astype(str).values
#     Xte = test_df.drop(columns=[LABEL_COL]).values
#     yte = test_df[LABEL_COL].astype(str).values

#     pipe = make_pipeline(
#         StandardScaler(with_mean=False),
#         RBFSampler(gamma=0.1, n_components=3000, random_state=42),
#         SGDClassifier(loss="hinge", max_iter=2000, random_state=42)
#     )

#     cv_used = 3
#     tuner = GridSearchCV(
#         pipe,
#         {
#             "rbfsampler__gamma":        [1, 0.1, 0.01, 0.001, 0.0001],
#             "rbfsampler__n_components": [2000, 4000],
#             "sgdclassifier__alpha":     [1e-4, 3e-4, 1e-3],
#         },
#         cv=cv_used,
#         scoring="accuracy",
#         return_train_score=True
#     )
#     tuner.fit(Xtr, ytr)

#     pred = tuner.best_estimator_.predict(Xte)

#     best_params = tuner.best_params_
#     best_score = tuner.best_score_
#     cv_results_df = pd.DataFrame(tuner.cv_results_)

#     labels = sorted(list(set(ytr) | set(yte)))
#     cm = confusion_matrix(yte, pred, labels=labels)

#     cm_df = pd.DataFrame(
#         cm,
#         columns=[f"Pred_{l}" for l in labels],
#         index=[f"Actual_{l}" for l in labels]
#     )

#     report_text = (
#         f"Model: RBF + SGDClassifier\n"
#         f"K-Fold CV: {cv_used}\n"
#         f"Best Params: {best_params}\n\n"
#         + classification_report(yte, pred, labels=labels, digits=4)
#     )

#     fig, ax = plt.subplots(figsize=(5, 4))
#     ax.imshow(cm, cmap="Blues")
#     plt.close(fig)

#     # Return 7 item
#     return (
#         report_text,
#         fig,
#         cm_df,
#         best_params,
#         best_score,
#         cv_results_df,
#         cv_used
#     )

STAGE_FUNCS = {
    1: ("Collect reviews", stage1_collect),
    2: ("Cleaning (no casefold)", stage2_clean),
    3: ("Case folding", stage3_casefold),
    4: ("Normalization", stage4_normalize),
    5: ("Tokenizing", stage5_tokenize),
    6: ("Labelling InSet", stage6_label_inset),
    7: ("TF-IDF summary", stage7_tfidf),
    8: ("Train/Test split", stage8_split),
    9: ("SVM Linear", stage9_svm_linear),
    10: ("SVM RBF", stage10_svm_rbf),
}

def run_stage(idx, args):
    title, fn = STAGE_FUNCS[idx]
    print(f"\n=== [Stage {idx}] {title} ===")
    if idx == 1:
        if not args.app: raise SystemExit("--app wajib untuk Stage 1 (collect).")
        return fn(app_input=args.app, out_dir=DATA_DIR, max_reviews=args.max_reviews,
                  page_size=args.page_size, lang=args.lang, country=args.country,
                  sleep_sec=args.sleep, from_date=DEFAULT_FROM_DATE, to_date=DEFAULT_TO_DATE)
    elif idx == 8:
        return fn(test_size=args.test_size, random_seed=args.seed)
    elif idx == 10:
        return fn()
    else:
        return fn()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pipeline Sentimen — Google Play Reviews (single file)")
    parser.add_argument("--stage", default="all", help="Tahap yang dijalankan: all | 1..10")
    parser.add_argument("--app", default=None, help="Package ID / URL Play Store / Nama app (wajib jika stage=1 atau all)")
    parser.add_argument("--max_reviews", type=int, default=0)
    parser.add_argument("--page_size", type=int, default=200)
    parser.add_argument("--lang", default="id")
    parser.add_argument("--country", default="id")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subsample", type=float, default=0.15, help="Fraksi data untuk tuning RBF (0.15 disarankan)")
    args = parser.parse_args()

    ensure_dir(DATA_DIR)

    if args.stage == "all":
        if not args.app:
            raise SystemExit("Saat --stage all, parameter --app wajib untuk Stage 1.")
        for i in range(1, 11):
            run_stage(i, args)
        print("\n🎉 Selesai: semua tahap telah dijalankan.")
    else:
        try:
            idx = int(args.stage)
            if idx < 1 or idx > 10: raise ValueError
        except ValueError:
            raise SystemExit("Nilai --stage tidak valid. Gunakan 'all' atau angka 1..10.")
        run_stage(idx, args)
        print(f"\n✅ Selesai Stage {idx}.")

if __name__ == "__main__":
    main()
# Aman di-import oleh Streamlit
