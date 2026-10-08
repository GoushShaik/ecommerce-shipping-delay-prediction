# =====================================================================
# E-Commerce Shipping Delay Prediction and Analytics Using Machine Learning
# Version 2: clean light-theme Gradio interface
# =====================================================================

# ------------------------- 1. IMPORTS -------------------------
import os, io, sys, inspect, warnings
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import gradio as gr
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
                     "text.color": "#111827", "axes.labelcolor": "#111827",
                     "xtick.color": "#111827", "ytick.color": "#111827"})

_base = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
DATA_PATH = os.path.join(_base, "Train.csv")
DT_PATH = os.path.join(_base, "decision_tree_model.pkl")
RF_PATH = os.path.join(_base, "random_forest_model.pkl")
COLS_PATH = os.path.join(_base, "model_columns.pkl")
CMP_PATH = os.path.join(_base, "model_comparison.pkl")

_required = {"Dataset": DATA_PATH, "Decision Tree model": DT_PATH, "Random Forest model": RF_PATH,
             "Model columns": COLS_PATH, "Model comparison": CMP_PATH}
_missing = [f"{n}: {p}" for n, p in _required.items() if not os.path.exists(p)]
if _missing:
    print("❌ Required files NOT found:")
    for m in _missing:
        print("   -", m)
    print("Please ensure all required dataset and model files are present in the repository root directory.")
    raise FileNotFoundError("Missing required files: " + "; ".join(_missing))
print("✅ All required files found.")

FLAGS = []   # start-up notes, printed in the console

# ------------------------- 3. LOAD DATASET -------------------------
TARGET = "Reached.on.Time_Y.N"
CAT_COLS = ["Warehouse_block", "Mode_of_Shipment", "Product_importance", "Gender"]
NUM_COLS = ["Customer_care_calls", "Customer_rating", "Cost_of_the_Product",
            "Prior_purchases", "Discount_offered", "Weight_in_gms"]
FEATURE_COLS = CAT_COLS + NUM_COLS
INT_COLS = ["Customer_care_calls", "Customer_rating", "Prior_purchases"]
STATUS = {0: "On Time", 1: "Delayed"}
LABELS = {"Warehouse_block": "Warehouse Block", "Mode_of_Shipment": "Mode of Shipment",
          "Customer_care_calls": "Customer Care Calls", "Customer_rating": "Customer Rating",
          "Cost_of_the_Product": "Cost of the Product", "Prior_purchases": "Prior Purchases",
          "Product_importance": "Product Importance", "Gender": "Gender",
          "Discount_offered": "Discount Offered", "Weight_in_gms": "Weight in Grams"}

try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    raise RuntimeError(f"Could not read the dataset at {DATA_PATH}: {e}")
_absent = [c for c in ["ID", TARGET] + FEATURE_COLS if c not in df.columns]
if _absent:
    raise ValueError(f"Dataset is missing columns: {_absent}. Found: {list(df.columns)}")
if not set(df[TARGET].unique()).issubset({0, 1}):
    raise ValueError("Target column must contain only 0 and 1.")

N_ROWS, N_COLS = df.shape
N_MISSING = int(df.isnull().sum().sum())
N_DUP = int(df.duplicated().sum())
N_ON = int((df[TARGET] == 0).sum())
N_DEL = int((df[TARGET] == 1).sum())
PCT_ON, PCT_DEL = 100 * N_ON / N_ROWS, 100 * N_DEL / N_ROWS
df["Delivery Status"] = df[TARGET].map(STATUS)

for _name, _exp, _got in [("rows", 10999, N_ROWS), ("columns", 12, N_COLS), ("missing values", 0, N_MISSING),
                          ("duplicate rows", 0, N_DUP), ("on-time count", 4436, N_ON), ("delayed count", 6563, N_DEL)]:
    if _exp != _got:
        FLAGS.append(f"Dataset {_name}: project notes say {_exp}, file has {_got}.")
print(f"✅ Dataset loaded: {N_ROWS} rows x {N_COLS} columns")

# ------------------------- 4. LOAD MODELS -------------------------
def safe_load(path, name):
    try:
        return joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"Could not load {name} from {path}: {e}")

DT_MODEL = safe_load(DT_PATH, "Decision Tree model")
RF_MODEL = safe_load(RF_PATH, "Random Forest model")
MODEL_COLUMNS = [str(c) for c in list(safe_load(COLS_PATH, "model columns"))]
RAW_CMP = safe_load(CMP_PATH, "model comparison")

if not isinstance(DT_MODEL, DecisionTreeClassifier):
    raise TypeError("decision_tree_model.pkl is not a DecisionTreeClassifier.")
for _n, _m in [("Decision Tree", DT_MODEL), ("Random Forest", RF_MODEL)]:
    if not hasattr(_m, "predict_proba") or not hasattr(_m, "feature_importances_"):
        raise TypeError(f"{_n} does not look like a fitted scikit-learn classifier.")
    if getattr(_m, "n_features_in_", len(MODEL_COLUMNS)) != len(MODEL_COLUMNS):
        raise ValueError(f"{_n} expects {_m.n_features_in_} features but model_columns.pkl has "
                         f"{len(MODEL_COLUMNS)}. Saved files do not belong together.")
    if not {0, 1}.issubset(set(_m.classes_)):
        raise ValueError(f"{_n} classes are {list(_m.classes_)}; expected 0 and 1.")
if len(MODEL_COLUMNS) != 15:
    FLAGS.append(f"Encoded features: project notes say 15, model_columns.pkl has {len(MODEL_COLUMNS)}.")

METRICS = ["Accuracy", "Precision", "Recall", "F1 Score"]
MODEL_NAMES = ["Decision Tree", "Random Forest"]
MODELS = {"Decision Tree": DT_MODEL, "Random Forest": RF_MODEL}


def normalize_comparison(raw):
    """DataFrame with rows = METRICS, columns = MODEL_NAMES, values as fractions (0-1)."""
    c = pd.DataFrame(raw).copy()
    c.columns = [str(x).strip() for x in c.columns]
    if "Metric" in c.columns:
        c = c.set_index("Metric")
    c.index = [str(x).strip() for x in c.index]
    if all(m in c.index for m in MODEL_NAMES):
        c = c.T
    key = lambda s: str(s).lower().replace("-", " ").replace("_", " ").strip()
    cl, il = {key(x): x for x in c.columns}, {key(x): x for x in c.index}
    try:
        cols = [cl[key(m)] for m in MODEL_NAMES]
        rows = [il[key(m)] for m in METRICS]
    except KeyError:
        raise ValueError("model_comparison.pkl has an unexpected layout.\n"
                         f"Columns found: {list(c.columns)}\nRows found: {list(c.index)}\n"
                         f"Expected metrics {METRICS} and models {MODEL_NAMES}.")
    out = c.loc[rows, cols].astype(float)
    out.index, out.columns = METRICS, MODEL_NAMES
    if out.values.max() > 1.0:
        out = out / 100.0
    return out

CMP = normalize_comparison(RAW_CMP)
ACC_TIE = CMP.loc["Accuracy", "Random Forest"] == CMP.loc["Accuracy", "Decision Tree"]
PRIMARY = "Random Forest" if CMP.loc["Accuracy", "Random Forest"] > CMP.loc["Accuracy", "Decision Tree"] else "Decision Tree"
if ACC_TIE:
    FLAGS.append("Saved accuracies are equal; the Decision Tree is primary by the tie rule.")
print(f"✅ Models loaded. Primary model (from saved accuracy): {PRIMARY}")

ONEHOT_MAP = {}
for _col in MODEL_COLUMNS:
    for _g in CAT_COLS:
        if _col.startswith(_g + "_"):
            ONEHOT_MAP[_col] = (_g, _col[len(_g) + 1:])
            break

# ------------------------- 5. RECONSTRUCT TEST SET -------------------------
X_all = pd.get_dummies(df[FEATURE_COLS], columns=CAT_COLS, drop_first=True)
_nat = set(X_all.columns)
if [c for c in MODEL_COLUMNS if c not in _nat] or [c for c in _nat if c not in set(MODEL_COLUMNS)]:
    FLAGS.append("Encoded dataset columns differ from model_columns.pkl; reindexed to the saved columns.")
X_all = X_all.reindex(columns=MODEL_COLUMNS, fill_value=0).astype(float)
y_all = df[TARGET].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
N_TRAIN, N_TEST = len(X_train), len(X_test)
if (N_TRAIN, N_TEST) != (8799, 2200):
    FLAGS.append(f"Train/test sizes: notes say 8799/2200, reconstructed {N_TRAIN}/{N_TEST}.")

# ------------------------- 6. METRICS (verification + confusion matrices) -------------------------
RECOMP, CONF = {}, {}
for _n, _m in MODELS.items():
    try:
        yp = _m.predict(X_test)
    except Exception as e:
        raise RuntimeError(f"Could not evaluate {_n} on the reconstructed test set: {e}")
    RECOMP[_n] = {"Accuracy": accuracy_score(y_test, yp),
                  "Precision": precision_score(y_test, yp, pos_label=1, zero_division=0),
                  "Recall": recall_score(y_test, yp, pos_label=1, zero_division=0),
                  "F1 Score": f1_score(y_test, yp, pos_label=1, zero_division=0)}
    CONF[_n] = confusion_matrix(y_test, yp, labels=[0, 1])
METRIC_MISMATCH = []
for _n in MODEL_NAMES:
    for _met in METRICS:
        d = abs(CMP.loc[_met, _n] - RECOMP[_n][_met]) * 100
        if d > 0.5:
            METRIC_MISMATCH.append(f"{_n} {_met}: saved {CMP.loc[_met,_n]*100:.2f}% vs recomputed {RECOMP[_n][_met]*100:.2f}%")
if METRIC_MISMATCH:
    FLAGS.append("Saved metrics differ from recomputed ones: " + "; ".join(METRIC_MISMATCH))

# ------------------------- 7. INPUT PREPROCESSING -------------------------
CAT_VALUES = {c: sorted(df[c].astype(str).unique()) for c in CAT_COLS}
_imp = [v for v in ["low", "medium", "high"] if v in CAT_VALUES["Product_importance"]]
CAT_VALUES["Product_importance"] = _imp or CAT_VALUES["Product_importance"]
NUM_RANGE = {c: (float(df[c].min()), float(df[c].max())) for c in NUM_COLS}


def validate_inputs(raw):
    clean, warns = {}, []
    for c in CAT_COLS:
        v = raw.get(c)
        if v is None or str(v) not in CAT_VALUES[c]:
            raise ValueError(f"Invalid value for {LABELS[c]}: {v!r}. Allowed: {CAT_VALUES[c]}")
        clean[c] = str(v)
    for c in NUM_COLS:
        v = raw.get(c)
        if v is None or v == "":
            raise ValueError(f"{LABELS[c]} is empty. Please enter a number.")
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"{LABELS[c]} must be a number (got {v!r}).")
        if not np.isfinite(v) or v < 0:
            raise ValueError(f"{LABELS[c]} must be a non-negative number (got {v}).")
        lo, hi = NUM_RANGE[c]
        if v < lo or v > hi:
            warns.append(f"{LABELS[c]} = {v:g} is outside the dataset range ({lo:g} to {hi:g}), "
                         f"so the models are working beyond the data they were trained on.")
        clean[c] = int(round(v)) if c in INT_COLS else v
    return clean, warns


def encode_input(clean):
    """One-hot encode, align to the saved training columns, fill missing with 0, keep exact order."""
    row = pd.DataFrame([clean])[FEATURE_COLS]
    enc = pd.get_dummies(row, columns=CAT_COLS)   # no drop_first: baseline columns are absent from MODEL_COLUMNS anyway
    enc = enc.reindex(columns=MODEL_COLUMNS, fill_value=0).astype(float)
    if list(enc.columns) != MODEL_COLUMNS:
        raise ValueError("Encoding mismatch: encoded input does not match the training columns.")
    return enc

# ------------------------- 8. PREDICTION FUNCTIONS -------------------------
def predict_one(model, X):
    proba = model.predict_proba(X)[0]
    cl = list(model.classes_)
    return {"pred": int(model.predict(X)[0]), "p_on": float(proba[cl.index(0)]), "p_del": float(proba[cl.index(1)])}


def run_both(raw):
    clean, warns = validate_inputs(raw)
    X = encode_input(clean)
    return {n: predict_one(m, X) for n, m in MODELS.items()}, X, clean, warns

# ------------------------- 9. HTML HELPERS (fixed dark-on-light colours) -------------------------
BLUE, TXT, SUB, BORDER, BG = "#1d4ed8", "#111827", "#374151", "#d1d5db", "#f3f4f6"
GREEN_T, GREEN_BG, RED_T, RED_BG = "#065f46", "#ecfdf5", "#991b1b", "#fef2f2"
AMBER_T, AMBER_BG = "#92400e", "#fffbeb"


def box(inner, bg=BG, border=BORDER, extra=""):
    return (f'<div style="background:{bg};border:1px solid {border};border-radius:10px;padding:16px 20px;'
            f'margin:8px 0;color:{TXT};{extra}">{inner}</div>')


def heading(text, sub=None):
    s = f'<div style="color:{SUB};font-size:15px;margin-top:4px;">{sub}</div>' if sub else ""
    return f'<div style="margin:18px 0 6px 0;"><div style="font-size:22px;font-weight:700;color:{TXT};">{text}</div>{s}</div>'


def note(text):
    return f'<div style="font-size:13px;color:{SUB};margin-top:8px;line-height:1.5;">{text}</div>'


def stat_card(value, label, sub=""):
    sub_html = f'<div style="font-size:12px;color:{SUB};">{sub}</div>' if sub else ""
    return (f'<div style="flex:1;min-width:150px;background:{BG};border:1px solid {BORDER};border-radius:10px;'
            f'padding:12px 14px;text-align:center;"><div style="font-size:26px;font-weight:700;color:{BLUE};">{value}</div>'
            f'<div style="font-size:14px;color:{TXT};">{label}</div>{sub_html}</div>')


def cards_row(cards):
    return '<div style="display:flex;flex-wrap:wrap;gap:12px;margin:10px 0;">' + "".join(cards) + "</div>"


def table(headers, rows, size="15px"):
    th = "".join(f'<th style="background:{BLUE};color:#ffffff;padding:9px 12px;text-align:left;border:1px solid {BORDER};">{h}</th>' for h in headers)
    body = ""
    for r in rows:
        body += "<tr>" + "".join(f'<td style="background:#ffffff;color:{TXT};padding:8px 12px;border:1px solid {BORDER};">{c}</td>' for c in r) + "</tr>"
    return f'<div style="overflow-x:auto;"><table style="border-collapse:collapse;width:100%;font-size:{size};"><tr>{th}</tr>{body}</table></div>'


def status_text(pred, upper=False):
    t = "Delayed" if int(pred) == 1 else "On Time"
    return t.upper() if upper else t


def status_chip(pred):
    if int(pred) == 1:
        return f'<span style="color:{RED_T};font-weight:700;">Delayed</span>'
    return f'<span style="color:{GREEN_T};font-weight:700;">On Time</span>'


def warn_box(msgs):
    if not msgs:
        return ""
    li = "".join(f"<li>{m}</li>" for m in msgs)
    return box(f"<b>Note</b><ul style='margin:6px 0 0 18px;'>{li}</ul>", AMBER_BG, AMBER_T)


def error_box(msg):
    return box(f"<b>Error</b><br>{str(msg)}".replace("\n", "<br>"), RED_BG, RED_T)


def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    img.load()
    return img

# ------------------------- 10. EXPLANATIONS: DECISION TREE PATH + FEATURE IMPORTANCE -------------------------
def explain_decision_tree(X, clean):
    tree = DT_MODEL.tree_
    ind = DT_MODEL.decision_path(X)
    path = list(ind.indices[ind.indptr[0]:ind.indptr[1]])
    steps = []
    for k, node in enumerate(path[:-1], start=1):
        fname = MODEL_COLUMNS[int(tree.feature[node])]
        thr = float(tree.threshold[node])
        left = path[k] == tree.children_left[node]
        if fname in ONEHOT_MAP:
            g, val = ONEHOT_MAP[fname]
            text = f"<b>{LABELS[g]}</b> is {'not ' if left else ''}'{val}' (your value: {clean[g]})"
        else:
            op = "is at most" if left else "is greater than"
            text = f"<b>{LABELS.get(fname, fname)}</b> {op} {thr:.2f} (your value: {clean[fname]:g})"
        steps.append(f'<li style="margin:4px 0;">{text}</li>')
    leaf = path[-1]
    v = np.array(tree.value[leaf][0], dtype=float)
    v = v / v.sum()
    cl = list(DT_MODEL.classes_)
    leaf_cls = cl[int(np.argmax(v))]
    inner = (f'<div style="font-weight:700;margin-bottom:6px;">Questions the tree asked about this shipment, in order:</div>'
             f'<div style="max-height:280px;overflow-y:auto;"><ol style="margin:0 0 0 20px;color:{TXT};">{"".join(steps)}</ol></div>'
             f'<div style="margin-top:10px;">The tree reached a final leaf and predicted: {status_chip(leaf_cls)} '
             f'(the {int(tree.n_node_samples[leaf])} training shipments in this leaf were '
             f'{v[cl.index(0)]*100:.1f}% On Time and {v[cl.index(1)]*100:.1f}% Delayed).</div>'
             + note("These features and thresholds are read directly from the trained Decision Tree."))
    return heading("Decision Tree Explanation", "The path this shipment followed through the tree.") + box(inner)


def grouped_importance(model):
    imp = pd.Series(model.feature_importances_, index=MODEL_COLUMNS)
    grouped = {}
    for col, v in imp.items():
        key = ONEHOT_MAP[col][0] if col in ONEHOT_MAP else col
        grouped[key] = grouped.get(key, 0.0) + float(v)
    s = pd.Series(grouped)
    s.index = [LABELS.get(i, i) for i in s.index]
    return s.sort_values(ascending=False)


def importance_bars(model, title, top=5):
    s = grouped_importance(model).head(top)
    mx = float(s.max()) if len(s) else 1.0
    rows = ""
    for name, v in s.items():
        w = max(2, int(100 * v / mx))
        rows += (f'<div style="display:flex;align-items:center;gap:10px;margin:6px 0;">'
                 f'<div style="width:170px;color:{TXT};font-size:14px;">{name}</div>'
                 f'<div style="flex:1;background:#e5e7eb;border-radius:4px;"><div style="width:{w}%;background:{BLUE};height:14px;border-radius:4px;"></div></div>'
                 f'<div style="width:55px;color:{TXT};font-size:14px;text-align:right;">{v:.3f}</div></div>')
    return f'<div style="flex:1;min-width:300px;"><div style="font-weight:700;margin-bottom:4px;">{title}</div>{rows}</div>'


def factors_html():
    inner = ('<div style="display:flex;flex-wrap:wrap;gap:24px;">' + importance_bars(DT_MODEL, "Decision Tree") +
             importance_bars(RF_MODEL, "Random Forest") + "</div>" +
             note("Top 5 features by importance. For categorical features, the importance values of their one-hot columns are added together. "
                  "Feature importance indicates which features the model relied on more during prediction. "
                  "It does not prove that a feature causes delivery delays."))
    return heading("Important Factors Considered by the Models") + box(inner)

# ------------------------- 11. RESULT + WHAT-IF -------------------------
def model_card(name, r):
    star = " (Primary)" if name == PRIMARY else ""
    return (f'<div style="flex:1;min-width:260px;background:#ffffff;border:1px solid {BORDER};border-radius:10px;padding:14px 16px;color:{TXT};">'
            f'<div style="font-size:16px;font-weight:700;color:{BLUE};">{name}{star}</div>'
            f'<div style="font-size:15px;margin-top:6px;">Prediction: {status_chip(r["pred"])}</div>'
            f'<div style="font-size:14px;margin-top:4px;">Model-estimated probability:</div>'
            f'<div style="font-size:14px;">On Time <b>{r["p_on"]*100:.1f}%</b> &nbsp;|&nbsp; Delayed <b>{r["p_del"]*100:.1f}%</b></div></div>')


def build_result(res, warns):
    final = res[PRIMARY]["pred"]
    if final == 1:
        bg, bd, fg, headline = RED_BG, RED_T, RED_T, "LIKELY TO BE DELAYED"
    else:
        bg, bd, fg, headline = GREEN_BG, GREEN_T, GREEN_T, "LIKELY TO REACH ON TIME"
    banner = (f'<div style="background:{bg};border:2px solid {bd};border-radius:12px;padding:20px;text-align:center;margin:8px 0;">'
              f'<div style="font-size:14px;color:{fg};">Prediction Result (from the primary model: {PRIMARY})</div>'
              f'<div style="font-size:30px;font-weight:800;color:{fg};margin-top:4px;">{headline}</div></div>')
    models = '<div style="display:flex;flex-wrap:wrap;gap:12px;">' + "".join(model_card(n, res[n]) for n in MODEL_NAMES) + "</div>"
    agree = res["Decision Tree"]["pred"] == res["Random Forest"]["pred"]
    agree_txt = "Both models agree on this prediction." if agree else "The models produced different predictions."
    agree_note = ("Agreement does not guarantee that the prediction is correct." if agree else
                  "The primary model's prediction is used as the final result.")
    why_primary = ("It has the higher accuracy on the test dataset." if not ACC_TIE
                   else "Both have equal accuracy on the test dataset, so the Decision Tree is used by default.")
    info = ('<div style="display:flex;flex-wrap:wrap;gap:12px;margin-top:12px;">'
            f'<div style="flex:1;min-width:260px;background:{BG};border:1px solid {BORDER};border-radius:10px;padding:12px 16px;color:{TXT};">'
            f'<div style="font-weight:700;">Model Agreement</div><div style="margin-top:4px;">{agree_txt}</div>{note(agree_note)}</div>'
            f'<div style="flex:1;min-width:260px;background:{BG};border:1px solid {BORDER};border-radius:10px;padding:12px 16px;color:{TXT};">'
            f'<div style="font-weight:700;">Primary Model</div><div style="margin-top:4px;">{PRIMARY}</div>'
            f'{note("The primary model is selected based on performance on the test dataset. " + why_primary)}</div></div>')
    return warn_box(warns) + heading("Prediction Result") + banner + models + info


PLACEHOLDER = box("Fill in the shipment details above and click <b>Analyze Shipment</b>. The result will appear here.")


def analyze_shipment(wb, mode, imp, gender, calls, rating, cost, prior, disc, weight):
    raw = {"Warehouse_block": wb, "Mode_of_Shipment": mode, "Product_importance": imp, "Gender": gender,
           "Customer_care_calls": calls, "Customer_rating": rating, "Cost_of_the_Product": cost,
           "Prior_purchases": prior, "Discount_offered": disc, "Weight_in_gms": weight}
    try:
        res, X, clean, warns = run_both(raw)
        return build_result(res, warns), factors_html(), explain_decision_tree(X, clean)
    except ValueError as e:
        return error_box(e), "", ""
    except Exception as e:
        return error_box(f"Prediction failed: {e}"), "", ""


MOD_KEYS = ["Customer_care_calls", "Cost_of_the_Product", "Prior_purchases", "Discount_offered", "Weight_in_gms"]


def analyze_whatif(wb, mode, imp, gender, calls, rating, cost, prior, disc, weight,
                   m_calls, m_cost, m_prior, m_disc, m_weight):
    orig = {"Warehouse_block": wb, "Mode_of_Shipment": mode, "Product_importance": imp, "Gender": gender,
            "Customer_care_calls": calls, "Customer_rating": rating, "Cost_of_the_Product": cost,
            "Prior_purchases": prior, "Discount_offered": disc, "Weight_in_gms": weight}
    mod = dict(orig)
    mod.update({"Customer_care_calls": m_calls, "Cost_of_the_Product": m_cost, "Prior_purchases": m_prior,
                "Discount_offered": m_disc, "Weight_in_gms": m_weight})
    try:
        r0, _, c0, w0 = run_both(orig)
        r1, _, c1, w1 = run_both(mod)
    except ValueError as e:
        return error_box(e)
    except Exception as e:
        return error_box(f"What-if analysis failed: {e}")
    changes = [f"{LABELS[k]}: {c0[k]:g} → {c1[k]:g}" for k in MOD_KEYS if c0[k] != c1[k]]
    chg = ("<b>Values changed:</b> " + "; ".join(changes)) if changes else "No values were changed, so both scenarios are identical."
    rows = []
    for n in MODEL_NAMES:
        o, m = r0[n], r1[n]
        rows.append([n + (" (Primary)" if n == PRIMARY else ""), status_chip(o["pred"]), f"{o['p_del']*100:.1f}%",
                     status_chip(m["pred"]), f"{m['p_del']*100:.1f}%", f"{(m['p_del']-o['p_del'])*100:+.1f} pp",
                     "Changed" if o["pred"] != m["pred"] else "Same"])
    inner = (f'<div style="margin-bottom:10px;">{chg}</div>' +
             table(["Model", "Original prediction", "Original Delayed prob.", "Modified prediction",
                    "Modified Delayed prob.", "Change", "Prediction"], rows, "14px") +
             f'<div style="margin-top:10px;">Final result from the primary model ({PRIMARY}): '
             f'{status_chip(r0[PRIMARY]["pred"])} → {status_chip(r1[PRIMARY]["pred"])}</div>' +
             note("pp = percentage points. This shows how the models' predictions change when the input values are changed. "
                  "It does not show that changing a value causes a shipment to be delayed or on time."))
    return warn_box(w0 + w1) + box(inner)

# ------------------------- 12. DATASET CHARTS -------------------------
C_ON, C_DEL = "#2563eb", "#f59e0b"   # on-time = blue, delayed = amber (colour-blind friendly, readable)


def chart_status():
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(["On Time", "Delayed"], [N_ON, N_DEL], color=[C_ON, C_DEL], width=0.55)
    for b, n, p in zip(bars, [N_ON, N_DEL], [PCT_ON, PCT_DEL]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + N_ROWS * 0.01, f"{n:,} ({p:.2f}%)", ha="center", fontsize=12)
    ax.set_ylim(0, max(N_ON, N_DEL) * 1.15)
    ax.set_xlabel("Delivery status")
    ax.set_ylabel("Number of shipments")
    ax.set_title("Delivery Status Distribution")
    return fig_to_image(fig)


def chart_box(col, title, ylabel):
    data = [df.loc[df[TARGET] == 0, col], df.loc[df[TARGET] == 1, col]]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bp = ax.boxplot(data, patch_artist=True, showmeans=True, widths=0.5,
                    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="#111827", markersize=8))
    for p, c in zip(bp["boxes"], [C_ON, C_DEL]):
        p.set_facecolor(c)
        p.set_alpha(0.6)
    for m in bp["medians"]:
        m.set_color("#111827")
    ax.set_xticklabels(["On Time", "Delayed"])
    ax.set_xlabel("Delivery status")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.text(0.98, 0.97, "Line = median, diamond = mean", transform=ax.transAxes, ha="right", va="top", fontsize=10, color="#374151")
    return fig_to_image(fig)


def chart_stacked(col, title, order=None):
    ct = pd.crosstab(df[col], df["Delivery Status"], normalize="index") * 100
    ct = ct.reindex(columns=["On Time", "Delayed"], fill_value=0)
    if order:
        ct = ct.reindex([o for o in order if o in ct.index])
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ct.plot(kind="bar", stacked=True, color=[C_ON, C_DEL], ax=ax, width=0.6, edgecolor="white")
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.1f%%", label_type="center", color="#111827", fontsize=11)
    ax.set_xlabel(LABELS.get(col, col))
    ax.set_ylabel("Percentage of shipments in each group")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.legend(title="Delivery status", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    plt.setp(ax.get_xticklabels(), rotation=0)
    return fig_to_image(fig)


def observations_html():
    g = df.groupby("Delivery Status")[["Discount_offered", "Weight_in_gms"]].mean()
    mp = (pd.crosstab(df["Mode_of_Shipment"], df[TARGET], normalize="index")[1] * 100).round(1)
    ip = (pd.crosstab(df["Product_importance"], df[TARGET], normalize="index")[1] * 100).round(1)
    dd, do = g.loc["Delayed", "Discount_offered"], g.loc["On Time", "Discount_offered"]
    wd, wo = g.loc["Delayed", "Weight_in_gms"], g.loc["On Time", "Weight_in_gms"]
    items = [
        f"Delayed shipments have a {'higher' if dd > do else 'lower'} average discount in this dataset ({dd:.2f} vs {do:.2f} for on-time shipments).",
        f"Delayed shipments have a {'higher' if wd > wo else 'lower'} average weight than on-time shipments in this dataset ({wd:.0f} g vs {wo:.0f} g).",
        "Shipment modes show differences in their observed delivery-status distribution (delayed share: " + ", ".join(f"{k} {v}%" for k, v in mp.items()) + ").",
        "Product importance groups also show observed differences (delayed share: " + ", ".join(f"{k} {v}%" for k, v in ip.items()) + ").",
    ]
    li = "".join(f'<li style="margin:6px 0;">{i}</li>' for i in items)
    return (heading("Key Observations") +
            box(f'<ul style="margin:0 0 0 20px;">{li}</ul>' +
                note("These are observed patterns and associations in this dataset. They do not prove that any feature causes a delay.")))

# ------------------------- 13. MODEL COMPARISON CONTENT -------------------------
def comparison_html():
    rows = []
    for met in METRICS:
        a, b = CMP.loc[met, "Decision Tree"], CMP.loc[met, "Random Forest"]
        fa, fb = f"{a*100:.2f}%", f"{b*100:.2f}%"
        if a > b:
            fa = f"<b>{fa} ✔</b>"
        elif b > a:
            fb = f"<b>{fb} ✔</b>"
        rows.append([met, fa, fb])
    out = heading("Decision Tree vs Random Forest", "Both models were trained on the same training data and evaluated on the same test data.")
    out += box(table(["Metric", "Decision Tree", "Random Forest"], rows) +
               note("✔ marks the higher value for that metric. Values are read from the saved comparison file."), "#ffffff")
    reason = (f"It has the higher Accuracy on the test dataset ({CMP.loc['Accuracy', PRIMARY]*100:.2f}%)." if not ACC_TIE
              else f"Both models have equal Accuracy ({CMP.loc['Accuracy', PRIMARY]*100:.2f}%), so the Decision Tree is used by default.")
    out += box(f'<div style="font-size:14px;color:{SUB};">Best Performing Model</div>'
               f'<div style="font-size:26px;font-weight:700;color:{BLUE};">{PRIMARY}</div>'
               f'<div style="margin-top:4px;">{reason}</div>' +
               note("The best model is chosen automatically from the saved Accuracy values and is used as the primary model on the prediction page."),
               "#eff6ff", BLUE)
    out += box("<b>What the metrics mean</b><ul style='margin:6px 0 0 20px;'>"
               "<li><b>Accuracy:</b> share of all test shipments predicted correctly.</li>"
               "<li><b>Precision:</b> of the shipments predicted Delayed, how many were really delayed.</li>"
               "<li><b>Recall:</b> of the shipments that were really delayed, how many the model found.</li>"
               "<li><b>F1 Score:</b> a balance between Precision and Recall.</li></ul>")
    if METRIC_MISMATCH:
        out += warn_box(["Recomputed metrics on the reconstructed test split differ from the saved file: " + "; ".join(METRIC_MISMATCH) +
                         ". The saved file may use a different averaging method or split. The saved values are shown above."])
    return out


def chart_confusion(name):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(CONF[name], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, annot_kws={"size": 16},
                xticklabels=["On Time", "Delayed"], yticklabels=["On Time", "Delayed"])
    ax.set_xlabel("Predicted status")
    ax.set_ylabel("Actual status")
    ax.set_title(f"Confusion Matrix: {name}")
    return fig_to_image(fig)


def confusion_note_html():
    rows = []
    for n in MODEL_NAMES:
        tn, fp, fn, tp = [int(v) for v in CONF[n].ravel()]
        rows.append([n, tn, fp, fn, tp])
    return box(table(["Model", "TN", "FP", "FN", "TP"], rows, "14px") +
               note(f"Computed on the same {N_TEST:,} test shipments, with Delayed as the positive class. "
                    "<b>TN</b>: on-time shipment predicted on time. <b>FP</b>: on-time shipment predicted delayed. "
                    "<b>FN</b>: delayed shipment predicted on time. <b>TP</b>: delayed shipment predicted delayed."))


def chart_importance(model, name, color):
    s = grouped_importance(model).sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.barh(s.index, s.values, color=color)
    for i, v in enumerate(s.values):
        ax.text(v + s.values.max() * 0.01, i, f"{v:.3f}", va="center", fontsize=11)
    ax.set_xlim(0, s.values.max() * 1.15)
    ax.set_xlabel("Feature importance (one-hot columns added together)")
    ax.set_ylabel("Feature")
    ax.set_title(f"{name}: Feature Importance")
    return fig_to_image(fig)

# ------------------------- 14. ABOUT PAGE CONTENT -------------------------
def flow_html():
    steps = ["Dataset", "Preprocessing", "EDA", "Feature Encoding", "Decision Tree", "Random Forest", "Model Comparison", "Web Application"]
    parts = []
    for i, s in enumerate(steps):
        parts.append(f'<div style="background:#ffffff;border:1px solid {BLUE};color:{TXT};border-radius:8px;padding:10px 14px;'
                     f'font-weight:600;text-align:center;min-width:130px;">{i+1}. {s}</div>')
        if i < len(steps) - 1:
            parts.append(f'<div style="color:{BLUE};font-size:20px;font-weight:700;">→</div>')
    return ('<div style="display:flex;flex-wrap:wrap;align-items:center;gap:8px;">' + "".join(parts) + "</div>")


def about_html():
    out = heading("Problem Statement")
    out += box("E-commerce companies ship a very large number of orders, and some of them arrive late. "
               "This project studies past shipment records to find patterns linked to on-time and delayed deliveries. "
               "It then trains two machine learning models to predict whether a new shipment is likely to be delayed.")
    out += heading("Dataset")
    out += box(f"<ul style='margin:0 0 0 20px;'><li><b>{N_ROWS:,}</b> records and <b>{N_COLS}</b> original columns "
               f"(<code>ID</code>, the target, and {len(FEATURE_COLS)} input features).</li>"
               f"<li>Target variable: <code>{TARGET}</code> &nbsp;→&nbsp; <b>0 = On Time</b>, <b>1 = Delayed</b>.</li>"
               f"<li><code>ID</code> was removed because it is only an identifier, not shipment information.</li>"
               f"<li>The 4 text columns (Warehouse block, Mode of shipment, Product importance, Gender) were one-hot encoded, giving "
               f"<b>{len(MODEL_COLUMNS)}</b> numeric input features.</li>"
               f"<li>Data was split 80% / 20% into <b>{N_TRAIN:,}</b> training and <b>{N_TEST:,}</b> testing records, "
               f"stratified so both parts keep the same on-time / delayed proportion.</li></ul>")
    out += heading("Methodology")
    out += box(flow_html(), "#ffffff")
    out += heading("Machine Learning Models")
    out += box("<ul style='margin:0 0 0 20px;'>"
               "<li><b>Decision Tree:</b> a tree-based classification model that is easy to interpret, because each prediction follows a path of simple questions.</li>"
               "<li><b>Random Forest:</b> an ensemble of multiple decision trees used for classification; their results are combined.</li></ul>"
               + note("Both models predict the same target and are compared using Accuracy, Precision, Recall and F1 Score."))
    out += heading("Limitations")
    out += box("<ul style='margin:0 0 0 20px;'>"
               "<li>Predictions are based on patterns in the available dataset.</li>"
               "<li>Model predictions are not guaranteed outcomes, and model probabilities are not necessarily calibrated.</li>"
               "<li>Feature importance does not imply causation.</li>"
               "<li>Performance depends on the dataset and the train/test split.</li>"
               "<li>The application predicts delivery status only, not an exact delivery date.</li></ul>", AMBER_BG, "#fcd34d")
    return out

# ------------------------- 15. GRADIO UI -------------------------
CSS = """
:root, .dark, body, body.dark, .gradio-container {
  color-scheme: light !important;
  --body-background-fill: #ffffff; --body-text-color: #111827; --body-text-color-subdued: #374151;
  --background-fill-primary: #ffffff; --background-fill-secondary: #f3f4f6;
  --block-background-fill: #ffffff; --block-border-color: #d1d5db; --block-label-background-fill: #f3f4f6;
  --block-label-text-color: #111827; --block-title-text-color: #111827; --block-info-text-color: #374151;
  --border-color-primary: #d1d5db; --input-background-fill: #ffffff; --input-border-color: #9ca3af;
  --panel-background-fill: #ffffff; --table-even-background-fill: #ffffff; --table-odd-background-fill: #f9fafb;
  --button-primary-background-fill: #1d4ed8; --button-primary-background-fill-hover: #1e40af;
  --button-primary-text-color: #ffffff; --button-secondary-background-fill: #f3f4f6;
  --button-secondary-text-color: #111827; --color-accent: #1d4ed8; --slider-color: #1d4ed8;
}
.gradio-container { background: #ffffff !important; color: #111827 !important; max-width: 1100px !important; margin: 0 auto !important; }
.gradio-container label, .gradio-container span, .gradio-container p, .gradio-container li, .gradio-container h1,
.gradio-container h2, .gradio-container h3 { color: #111827; }
.gradio-container input, .gradio-container textarea, .gradio-container select { color: #111827 !important; background: #ffffff !important; }
.gradio-container button.primary, .gradio-container button.lg.primary { background: #1d4ed8 !important; color: #ffffff !important; font-size: 18px !important; font-weight: 700 !important; }
.gradio-container button.primary * { color: #ffffff !important; }
.gradio-container button.secondary { background: #f3f4f6 !important; color: #111827 !important; border: 1px solid #9ca3af !important; }
.gradio-container .tab-nav button, .gradio-container [role="tab"] { color: #111827 !important; font-size: 16px !important; font-weight: 600 !important; }
.gradio-container [role="tab"][aria-selected="true"] { color: #1d4ed8 !important; border-bottom: 3px solid #1d4ed8 !important; }
footer { display: none !important; }
"""
FORCE_LIGHT_JS = "() => { document.body.classList.remove('dark'); document.documentElement.classList.remove('dark'); }"
THEME = gr.themes.Base(primary_hue="blue", neutral_hue="slate")
UI_OPTS = {"theme": THEME, "css": CSS, "js": FORCE_LIGHT_JS}
_launch_params = set(inspect.signature(gr.Blocks.launch).parameters)
BLOCKS_KW = {k: v for k, v in UI_OPTS.items() if k not in _launch_params}
LAUNCH_KW = {k: v for k, v in UI_OPTS.items() if k in _launch_params}


def num_default(col):
    return int(round(float(df[col].median())))


def slider_kw(col):
    lo, hi = NUM_RANGE[col]
    return dict(minimum=int(np.floor(lo)), maximum=int(np.ceil(hi)), value=num_default(col), step=1)


def number_kw(col, info):
    lo, hi = NUM_RANGE[col]
    return dict(value=num_default(col), minimum=0, info=f"{info} Dataset range: {lo:g} to {hi:g}.")


def cat_default(col):
    m = str(df[col].mode()[0])
    return m if m in CAT_VALUES[col] else CAT_VALUES[col][0]


HEADER = (f'<div style="padding:8px 0 4px 0;"><div style="font-size:30px;font-weight:800;color:{TXT};">E-Commerce Shipping Delay Prediction</div>'
          f'<div style="font-size:16px;color:{SUB};margin-top:4px;">Predict whether an e-commerce shipment is likely to reach on time using Machine Learning.</div></div>'
          + cards_row([stat_card(f"{N_ROWS:,}", "Dataset Records"),
                       stat_card(f"{N_COLS}", "Original Columns", f"{len(FEATURE_COLS)} used as inputs"),
                       stat_card("2", "Machine Learning Models", "Decision Tree, Random Forest")]))

# Pre-compute charts once
IMG_STATUS = chart_status()
IMG_DISC = chart_box("Discount_offered", "Discount Offered vs Delivery Status", "Discount offered")
IMG_WEIGHT = chart_box("Weight_in_gms", "Weight in Grams vs Delivery Status", "Weight (grams)")
IMG_MODE = chart_stacked("Mode_of_Shipment", "Shipment Mode vs Delivery Status")
IMG_IMP = chart_stacked("Product_importance", "Product Importance vs Delivery Status", ["low", "medium", "high"])
IMG_CM = {n: chart_confusion(n) for n in MODEL_NAMES}
IMG_FI_DT = chart_importance(DT_MODEL, "Decision Tree", "#1d4ed8")
IMG_FI_RF = chart_importance(RF_MODEL, "Random Forest", "#0f766e")

with gr.Blocks(title="E-Commerce Shipping Delay Prediction", **BLOCKS_KW) as demo:
    with gr.Tabs():
        # ---------- PAGE 1: PREDICT SHIPMENT ----------
        with gr.Tab("Predict Shipment"):
            gr.HTML(HEADER)
            gr.HTML(heading("Enter Shipment Details", "Step 1: fill in the values below. Step 2: click Analyze Shipment."))
            with gr.Row():
                with gr.Column():
                    gr.HTML(f'<div style="font-weight:700;font-size:17px;color:{BLUE};">Shipment Information</div>')
                    in_wb = gr.Dropdown(CAT_VALUES["Warehouse_block"], value=cat_default("Warehouse_block"), label="Warehouse Block")
                    in_mode = gr.Dropdown(CAT_VALUES["Mode_of_Shipment"], value=cat_default("Mode_of_Shipment"), label="Mode of Shipment")
                    in_imp = gr.Dropdown(CAT_VALUES["Product_importance"], value=cat_default("Product_importance"), label="Product Importance")
                    in_gender = gr.Dropdown(CAT_VALUES["Gender"], value=cat_default("Gender"), label="Gender")
                with gr.Column():
                    gr.HTML(f'<div style="font-weight:700;font-size:17px;color:{BLUE};">Customer &amp; Product Information</div>')
                    in_calls = gr.Slider(label="Customer Care Calls", **slider_kw("Customer_care_calls"))
                    in_rating = gr.Slider(label="Customer Rating (1 = lowest, 5 = highest)", **slider_kw("Customer_rating"))
                    in_prior = gr.Slider(label="Prior Purchases", **slider_kw("Prior_purchases"))
                    in_cost = gr.Number(label="Cost of the Product", **number_kw("Cost_of_the_Product", "Product price."))
                    in_disc = gr.Number(label="Discount Offered", **number_kw("Discount_offered", "Discount given on the order."))
                    in_weight = gr.Number(label="Weight in Grams", **number_kw("Weight_in_gms", "Package weight."))
            btn_analyze = gr.Button("Analyze Shipment", variant="primary", size="lg")

            out_result = gr.HTML(PLACEHOLDER)
            out_factors = gr.HTML("")
            out_path = gr.HTML("")

            main_inputs = [in_wb, in_mode, in_imp, in_gender, in_calls, in_rating, in_cost, in_prior, in_disc, in_weight]
            btn_analyze.click(analyze_shipment, inputs=main_inputs, outputs=[out_result, out_factors, out_path])

            gr.HTML(heading("What-If Analysis", "Change selected shipment values and observe how the trained models' predictions change. "
                            "The original scenario uses the values entered above."))
            with gr.Row():
                m_weight = gr.Number(label="Modified Weight in Grams", **number_kw("Weight_in_gms", "New weight."))
                m_disc = gr.Number(label="Modified Discount Offered", **number_kw("Discount_offered", "New discount."))
                m_cost = gr.Number(label="Modified Cost of the Product", **number_kw("Cost_of_the_Product", "New cost."))
            with gr.Row():
                m_prior = gr.Slider(label="Modified Prior Purchases", **slider_kw("Prior_purchases"))
                m_calls = gr.Slider(label="Modified Customer Care Calls", **slider_kw("Customer_care_calls"))
            with gr.Row():
                btn_whatif = gr.Button("Analyze Modified Scenario", variant="primary")
                btn_reset = gr.Button("Reset to original values", variant="secondary")
            out_whatif = gr.HTML("")
            btn_whatif.click(analyze_whatif, inputs=main_inputs + [m_calls, m_cost, m_prior, m_disc, m_weight], outputs=out_whatif)
            btn_reset.click(lambda a, b, c, d, e: (a, b, c, d, e),
                            inputs=[in_calls, in_cost, in_prior, in_disc, in_weight],
                            outputs=[m_calls, m_cost, m_prior, m_disc, m_weight])

        # ---------- PAGE 2: DATASET INSIGHTS ----------
        with gr.Tab("Dataset Insights"):
            gr.HTML(heading("Dataset Overview", "Calculated from Train.csv when the application starts."))
            gr.HTML(cards_row([stat_card(f"{N_ROWS:,}", "Records"), stat_card(f"{N_COLS}", "Features (columns)"),
                               stat_card(f"{N_MISSING}", "Missing Values"), stat_card(f"{N_DUP}", "Duplicate Rows")]))
            gr.HTML(heading("Delivery Status", "0 = On Time, 1 = Delayed"))
            gr.HTML(cards_row([stat_card(f"{N_ON:,}", "On Time", f"{PCT_ON:.2f}%"), stat_card(f"{N_DEL:,}", "Delayed", f"{PCT_DEL:.2f}%")]))
            gr.Image(IMG_STATUS, show_label=False, interactive=False)
            gr.HTML(heading("Visualizations", "How the main features differ between on-time and delayed shipments."))
            with gr.Row():
                gr.Image(IMG_DISC, show_label=False, interactive=False)
                gr.Image(IMG_WEIGHT, show_label=False, interactive=False)
            with gr.Row():
                gr.Image(IMG_MODE, show_label=False, interactive=False)
                gr.Image(IMG_IMP, show_label=False, interactive=False)
            gr.HTML(observations_html())

        # ---------- PAGE 3: MODEL COMPARISON ----------
        with gr.Tab("Model Comparison"):
            gr.HTML(heading("Model Comparison"))
            gr.HTML(comparison_html())
            gr.HTML(heading("Confusion Matrices", "Rows show the actual status, columns show the predicted status."))
            with gr.Row():
                gr.Image(IMG_CM["Decision Tree"], show_label=False, interactive=False)
                gr.Image(IMG_CM["Random Forest"], show_label=False, interactive=False)
            gr.HTML(confusion_note_html())
            gr.HTML(heading("Feature Importance", "Which features each model relied on more. This does not prove cause and effect."))
            with gr.Row():
                gr.Image(IMG_FI_DT, show_label=False, interactive=False)
                gr.Image(IMG_FI_RF, show_label=False, interactive=False)

        # ---------- PAGE 4: ABOUT PROJECT ----------
        with gr.Tab("About Project"):
            gr.HTML(about_html())

# ------------------------- 16. LAUNCH -------------------------
if __name__ == "__main__":
    if FLAGS:
        print("\n⚠️ Start-up notes:")
        for f in FLAGS:
            print("   -", f)
    server_port = int(os.environ.get("PORT", 7860))
    print(f"\n🚀 Launching application on 0.0.0.0:{server_port}...")
    demo.launch(server_name="0.0.0.0", server_port=server_port, **LAUNCH_KW)