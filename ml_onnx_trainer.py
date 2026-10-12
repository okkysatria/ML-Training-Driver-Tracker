"""Pelatihan NumPy (MLP 28-16-1 / GLM) yang benar: pembersihan data, prior anti-bocor (leave-one-day-out),
split kronologis per HARI (latih/validasi/uji), early stopping, pembanding baseline, ekspor ONNX."""
import json, datetime as dt
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import feature_engine as fe

MODELS = Path(__file__).parent / "models"
MODELS.mkdir(exist_ok=True)
DEFAULT = MODELS / "default_surabaya.onnx"
NF, MIN_CELL = len(fe.FEATURES), 3
NB = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)]


class Priors:
    """Statistik historis per sel. get(loo=True) membuang hari yang sama -> baris latih tidak 'mengintip' labelnya sendiri."""
    def __init__(self, recs):
        days = {r["when"].date() for r in recs}; self.D = len(days); self.nw = Counter(d.weekday() for d in days)
        self.tot, self.cd, self.slot, self.cds, self.dow, self.inn, self.ind = (Counter() for _ in range(7))
        for r in recs:
            c, d = r["cell"], r["when"].date(); s = fe.slot_of(r["when"].hour + r["when"].minute / 60)
            self.tot[c] += 1; self.cd[c, d] += 1; self.slot[c, s] += 1; self.cds[c, d, s] += 1; self.dow[c, d.weekday()] += 1
            self.inn[r["drop"]] += 1; self.ind[r["drop"], d] += 1

    def get(self, c, when, loo=False):
        d, x = when.date(), int(loo); s, w = fe.slot_of(when.hour + when.minute / 60), when.weekday()
        nd, nw = max(self.D - x, 1), max(self.nw.get(w, 0) - x, 1)
        t, tt, tw, ti = self.tot[c], self.slot[c, s], self.dow[c, w], self.inn[c]
        nb = sum(self.tot[c[0] + i, c[1] + j] for i, j in NB)
        if loo:
            t -= self.cd[c, d]; tt -= self.cds[c, d, s]; tw -= self.cd[c, d]; ti -= self.ind[c, d]
            nb -= sum(self.cd[(c[0] + i, c[1] + j), d] for i, j in NB)
        return (t / nd, ti / nd, tt / nd, tw / nw, nb / nd)

    def table(self, min_count=MIN_CELL):
        cells = [c for c, v in self.tot.items() if v >= min_count]; D = max(self.D, 1); k = lambda c: f"{c[0]},{c[1]}"
        return dict(cell_rate={k(c): self.tot[c] / D for c in cells}, cell_inflow={k(c): self.inn[c] / D for c in cells},
                    cell_slot={k(c): [self.slot[c, s] / D for s in range(5)] for c in cells},
                    cell_dow={k(c): [self.dow[c, w] / max(self.nw.get(w, 0), 1) for w in range(7)] for c in cells},
                    cell_nb={k(c): sum(self.tot[c[0] + i, c[1] + j] for i, j in NB) / D for c in cells})


def _sig(x): return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def _fwd(P, X, mode, h):
    Z = X @ P[0] + P[1] if h else None
    A = np.maximum(Z, 0) if h else X
    O = A @ P[2] + P[3]
    return Z, A, O, (_sig(O) if mode == "prob" else np.maximum(O, 0))


def _loss(P, X, y, mode, h):
    o = _fwd(P, X, mode, h)[3].ravel()
    if mode == "prob": o = np.clip(o, 1e-7, 1 - 1e-7); return float(-np.mean(y * np.log(o) + (1 - y) * np.log(1 - o)))
    return float(np.mean((o - y) ** 2))


def _train(X, y, Xv, yv, h, epochs, mode, prog, seed=0, patience=10, init=None):
    rng = np.random.default_rng(seed); n, d = X.shape
    P = ([rng.normal(0, np.sqrt(2 / d), (d, h)), np.zeros(h), rng.normal(0, np.sqrt(1 / h), (h, 1)), np.zeros(1)] if h
         else [None, None, rng.normal(0, .1, (d, 1)), np.zeros(1)])
    if init: P = [None if a is None else np.array(a, np.float64) for a in init]  # warm start: lanjut dari bobot model lama
    idx = [i for i in range(4) if P[i] is not None]; m = {i: np.zeros_like(P[i]) for i in idx}; v = {i: np.zeros_like(P[i]) for i in idx}
    yc = y.reshape(-1, 1); t, best, bestP, best_ep, wait = 0, 1e18, None, 1, 0
    for ep in range(epochs):
        lr = 0.01 * 0.98 ** ep; perm = rng.permutation(n)
        for s in range(0, n, 128):
            b = perm[s:s + 128]; xb, yb = X[b], yc[b]; Z, A, O, out = _fwd(P, xb, mode, h)
            dO = (out - yb) / len(b) if mode == "prob" else 2 * (out - yb) * (O > 0) / len(b)
            g = {2: A.T @ dO + 1e-4 * P[2], 3: dO.sum(0)}
            if h: dZ = (dO @ P[2].T) * (Z > 0); g[0] = xb.T @ dZ + 1e-4 * P[0]; g[1] = dZ.sum(0)
            t += 1
            for i in idx:  # Adam
                m[i] = .9 * m[i] + .1 * g[i]; v[i] = .999 * v[i] + .001 * g[i] ** 2
                P[i] -= lr * (m[i] / (1 - .9 ** t)) / (np.sqrt(v[i] / (1 - .999 ** t)) + 1e-8)
        if Xv is not None:  # early stopping pada data VALIDASI (hari-hari yang tidak dipakai belajar)
            vl = _loss(P, Xv, yv, mode, h)
            if vl < best - 1e-5: best, bestP, best_ep, wait = vl, [None if p is None else p.copy() for p in P], ep + 1, 0
            else:
                wait += 1
                if wait >= patience: break
            prog((ep + 1) / epochs, f"epoch {ep + 1}/{epochs} · loss validasi {vl:.3f}")
        else: prog((ep + 1) / epochs, f"epoch {ep + 1}/{epochs}")
    if Xv is None: bestP, best_ep = [None if p is None else p.copy() for p in P], ep + 1  # pelatihan akhir: jumlah epoch tetap
    return bestP, best_ep, ep + 1


def _export(P, h, mode, path):
    import onnx
    from onnx import helper as H, TensorProto as T, numpy_helper as N
    f32 = lambda a, n: N.from_array(np.asarray(a, np.float32), n); nodes, inits, cur = [], [], "input"
    if h:
        inits += [f32(P[0], "W1"), f32(P[1], "b1")]
        nodes += [H.make_node("MatMul", [cur, "W1"], ["z1"]), H.make_node("Add", ["z1", "b1"], ["z1b"]), H.make_node("Relu", ["z1b"], ["a1"])]; cur = "a1"
    inits += [f32(P[2], "W2"), f32(P[3], "b2")]
    nodes += [H.make_node("MatMul", [cur, "W2"], ["z2"]), H.make_node("Add", ["z2", "b2"], ["z2b"]), H.make_node("Sigmoid" if mode == "prob" else "Relu", ["z2b"], ["output"])]
    g = H.make_graph(nodes, "driver_demand", [H.make_tensor_value_info("input", T.FLOAT, [None, NF])], [H.make_tensor_value_info("output", T.FLOAT, [None, 1])], inits)
    mdl = H.make_model(g, opset_imports=[H.make_opsetid("", 13)]); mdl.ir_version = 8; onnx.checker.check_model(mdl); onnx.save(mdl, str(path))


def _build(recs, active, pri, opts, rng, neg, loo):
    """Baris = (tanggal, jam, sel). Positif = ada order; negatif = sel aktif lain pada (tanggal, jam) yang sama."""
    groups, aset = defaultdict(Counter), set(active)
    for r in recs:
        if r["cell"] in aset: groups[r["when"].date(), r["when"].hour][r["cell"]] += 1
    X, y, G, B = [], [], [], []
    for gi, ((d, hr), cnt) in enumerate(sorted(groups.items())):
        when = dt.datetime(d.year, d.month, d.day, hr, 30); pool = [c for c in active if c not in cnt]
        if neg and len(pool) > neg * len(cnt): pool = [pool[i] for i in rng.choice(len(pool), neg * len(cnt), replace=False)]
        for c, k in [(c, cnt[c]) for c in cnt] + [(c, 0) for c in pool]:
            p = pri.get(c, when, loo); X.append(fe.extract(c[0], c[1], when, p, opts)); y.append(k); G.append(gi); B.append((p[0], p[2], d.toordinal()))
    return np.array(X, np.float32), np.array(y, np.float32), np.array(G), np.array(B, np.float32)


NAME = {0: "GLM", 8: "MLP-8", 16: "MLP-16", 32: "MLP-32"}
nm = lambda h: NAME.get(h, f"MLP-{h}")


def _hits(score, y, G):
    """1/0 per grup (tanggal,jam) yang punya order: apakah ada order di 5 area teratas."""
    by = defaultdict(list)
    for i, g in enumerate(G): by[g].append(i)
    out = []
    for g in sorted(by):
        ix = np.array(by[g])
        if (y[ix] > 0).any(): out.append(float((y[ix[np.argsort(-score[ix], kind="stable")[:5]]] > 0).any()))
    return np.array(out)


def _gdays(y, G, day):
    """Hari (ordinal) untuk tiap grup yang punya order; urutannya sama dengan _hits."""
    first, pos = {}, set()
    for i, g in enumerate(G):
        first.setdefault(g, day[i])
        if y[i] > 0: pos.add(g)
    return np.array([first[g] for g in sorted(first) if g in pos])


def _cboot(v, days, rng, n=1000):
    """Bootstrap per HARI (jam-jam dalam satu hari saling berkaitan, jadi hari yang diacak, bukan jam)."""
    u, inv = np.unique(days, return_inverse=True); s = np.bincount(inv, weights=v); c = np.bincount(inv).astype(float)
    idx = rng.integers(0, len(u), (n, len(u))); return s[idx].sum(1) / c[idx].sum(1)


def _top5(score, y, G):
    h = _hits(score, y, G); return float(h.mean()) if len(h) else 0.0


def _evaluate(out, y, G, B, mode):
    out = out.ravel(); p = out if mode == "prob" else 1 - np.exp(-out); yb = (y > 0).astype(int)
    mi = np.arange(len(yb)); pm, ym = np.clip(p[mi], 1e-7, 1 - 1e-7), yb[mi]; npos, nneg = ym.sum(), (1 - ym).sum()
    # ROC-AUC must assign average ranks to tied scores; arbitrary tie order biases it.
    order = np.argsort(pm, kind="mergesort"); ranked = pm[order]; ranks = np.empty(len(pm), np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and ranked[j] == ranked[i]: j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2
        i = j
    auc = ((ranks[ym == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else .5
    pr = pm >= .5; tp = (pr & (ym == 1)).sum(); prec, rec = tp / max(pr.sum(), 1), tp / max(npos, 1)
    return {k: round(float(v), 4) for k, v in dict(
        top5_hit_rate=_top5(p, y, G), top5_baseline_popular=_top5(B[:, 0], y, G), top5_baseline_slot=_top5(B[:, 1] + B[:, 0] * 1e-3, y, G),
        roc_auc=auc, f1=2 * prec * rec / max(prec + rec, 1e-9), precision=prec, recall=rec,
        log_loss=-np.mean(ym * np.log(pm) + (1 - ym) * np.log(1 - pm)), brier=np.mean((pm - ym) ** 2), mae=np.mean(np.abs(out[mi] - y[mi]))).items()}


def _folds(days, cv):
    """Validasi silang berurutan (expanding window): latih di masa lalu, uji di blok hari berikutnya. Tanpa CV / data pendek: 1 pembagian 70/15/15."""
    n = len(days)
    if cv and n >= 28:
        b = max(int(n * .15), 3); out = []
        for j in range(3):
            t1 = n - (2 - j) * b; t0 = t1 - b; v0 = t0 - b; out.append((days[:v0], days[v0:t0], days[t0:t1]))
        return out
    i1 = max(int(n * .7), 3); i2 = min(max(i1 + 1, int(n * .85)), n - 1); return [(days[:i1], days[i1:i2], days[i2:])]


def train_model(orders, cfg, prog=lambda p, m: None):
    km, mode = float(cfg.get("km", 1.0)), cfg.get("target", "prob"); epochs = int(cfg.get("epochs", 60))
    opts = {"calendar": cfg.get("calendar", True), "payday": cfg.get("payday", True)}
    prog(2, "Membersihkan data…"); recs, removed = fe.clean_orders(orders, km); lm = int(cfg.get("last_months", 0))
    if lm and recs:  # opsional: hanya pakai N bulan terakhir (pola musiman lama dibuang)
        cut = recs[-1]["when"] - dt.timedelta(days=30 * lm); n0 = len(recs); recs = [r for r in recs if r["when"] >= cut]
        if n0 > len(recs): removed[f"lebih tua dari {lm} bulan"] = n0 - len(recs)
    days = sorted({r["when"].date() for r in recs}); n = len(days)
    if len(recs) < 100 or n < 7: raise ValueError(f"Data bersih: {len(recs)} order, {n} hari. Minimal 100 order dan 7 hari berbeda. Dibuang: {removed or 'tidak ada'}")
    init, warm, parent, wmn, wmx, hw = None, False, cfg.get("parent"), None, None, None
    if cfg.get("init_from"):  # lanjutkan dari model lama bila fitur & target sama; kalau tidak, latih dari awal
        try:
            ip = Path(cfg["init_from"]); om = json.loads(ip.with_name(ip.stem + "_metadata.json").read_text("utf-8")); w0 = _load_weights(ip)
            if om.get("features") == fe.FEATURES and om.get("target") == mode:
                init, warm, hw = w0, True, (0 if w0[0] is None else int(w0[0].shape[1])); wmn, wmx = np.array(om["scaler"]["min"], np.float32), np.array(om["scaler"]["max"], np.float32)
        except Exception: pass
    cands = [hw] if warm else {"auto": [0, 8, 16, 32], "mlp": [16], "glm": [0]}.get(cfg.get("algo", "auto"), [0, 8, 16, 32])
    folds = _folds(days, bool(cfg.get("cv", True)) and not warm); nsteps = len(folds) * (1 + len(cands)) + 1; st = [0]; rng = np.random.default_rng(1)

    def nxt(label):  # sub-callback progres untuk satu langkah
        a, w = 5 + 83 * st[0] / nsteps, 83 / nsteps; st[0] += 1; return lambda f, m: prog(a + w * f, f"{label} · {m}")

    sel = lambda D: [r for r in recs if r["when"].date() in D]
    outs, ys, Gs, Bs, beps = {h: [] for h in cands}, [], [], [], {h: [] for h in cands}
    vouts, vys, vGs, vBs = {h: [] for h in cands}, [], [], []
    goff = vgoff = n_te = cov_ok = 0; Pw, ran = None, 0
    for fi, (dtr, dva, dte) in enumerate(folds):
        lab = f"lipatan {fi + 1}/{len(folds)}" if len(folds) > 1 else "pembagian data"; nxt(lab)(0, "membangun fitur anti-bocor…")
        tr_r, va_r, te_r = sel(set(dtr)), sel(set(dva)), sel(set(dte)); P0 = Priors(tr_r); active = sorted(c for c, v in P0.tot.items() if v >= MIN_CELL)
        if len(active) < 3: raise ValueError("Terlalu sedikit area yang punya ≥3 order di data latih.")
        Xtr, ytr, _, _ = _build(tr_r, active, P0, opts, rng, 2, True)
        Xva, yva, Gva, Bva = _build(va_r, active, P0, opts, rng, 2, False)
        Xte, yte, Gte, Bte = _build(te_r, active, P0, opts, rng, None, False)
        if not len(Xva) or not len(Xte): raise ValueError("Hari validasi/uji tidak punya order di area yang dikenal. Tambah data.")
        if mode == "prob": ytr, yva = (ytr > 0).astype(np.float32), (yva > 0).astype(np.float32)
        mn, mx = (wmn, wmx) if warm else (Xtr.min(0), Xtr.max(0)); S = lambda X, mn=mn, mx=mx: (X - mn) / (mx - mn + 1e-9)
        for h in cands:
            P, bep, ran = _train(S(Xtr), ytr, S(Xva), yva, h, epochs, mode, nxt(f"{lab} · {nm(h)}"), init=init); Pw = P
            vouts[h].append(_fwd(P, S(Xva), mode, h)[3].ravel())
            outs[h].append(_fwd(P, S(Xte), mode, h)[3].ravel()); beps[h].append(bep)
        vys.append(yva); vGs.append(Gva + vgoff); vBs.append(Bva); vgoff += int(Gva.max()) + 1
        ys.append(yte); Gs.append(Gte + goff); Bs.append(Bte); goff += int(Gte.max()) + 1
        if fi == len(folds) - 1:
            n_te, n_tr, n_va = len(te_r), len(tr_r), len(va_r)
            aset = set(active); cov_ok = sum(r["cell"] in aset for r in te_r)
    # Choose architecture and baseline on validation only; test folds are held back for final metrics.
    yv_p, Gv_p, Bv_p = np.concatenate(vys), np.concatenate(vGs), np.concatenate(vBs)
    o_v = {h: np.concatenate(vouts[h]) for h in cands}
    gdv = _gdays(yv_p, Gv_p, Bv_p[:, 2]); hitv = {h: _hits(o_v[h], yv_p, Gv_p) for h in cands}
    hbv_pop = _hits(Bv_p[:, 0], yv_p, Gv_p)
    hbv_slot = _hits(Bv_p[:, 1] + Bv_p[:, 0] * 1e-3, yv_p, Gv_p)
    use_popular_baseline = hbv_pop.mean() >= hbv_slot.mean()
    hbest = max(cands, key=lambda h: hitv[h].mean()); bm = hitv[hbest].mean(); se = float(np.std(_cboot(hitv[hbest], gdv, rng)))
    chosen = next(h for h in sorted(cands) if hitv[h].mean() >= bm - se)
    # Keep the final chronological test block completely separate from selection.
    y_p, G_p, B_p, o_p = ys[-1], Gs[-1], Bs[-1], {h: outs[h][-1] for h in cands}
    gd = _gdays(y_p, G_p, B_p[:, 2]); hit = _hits(o_p[chosen], y_p, G_p)
    hb = _hits(B_p[:, 0], y_p, G_p) if use_popular_baseline else _hits(B_p[:, 1] + B_p[:, 0] * 1e-3, y_p, G_p)
    metrics = _evaluate(o_p[chosen], y_p, G_p, B_p, mode); d = hit - hb; lo, hi = np.percentile(_cboot(d, gd, rng), [5, 95])
    metrics.update({k: round(float(v), 4) for k, v in dict(top5_baseline_best=hb.mean(), lift_vs_best_baseline=d.mean(), lift_ci_low=lo, lift_ci_high=hi, coverage=cov_ok / max(n_te, 1)).items()})
    # --- model akhir: dilatih ulang pada SEMUA data (termasuk yang terbaru) dengan jumlah epoch dari validasi
    Pall = Priors(recs)
    act = sorted(c for c, v in Pall.tot.items() if v >= MIN_CELL); Xr, yr, _, _ = _build(recs, act, Pall, opts, rng, 2, True)
    if mode == "prob": yr = (yr > 0).astype(np.float32)
    if warm: mn, mx = wmn, wmx
    else: mn, mx = Xr.min(0), Xr.max(0)
    E = int(min(max(np.mean(beps[chosen]) * 1.2 + 1, 10), epochs))
    P, _, ran = _train((Xr - mn) / (mx - mn + 1e-9), yr, None, None, chosen, E, mode, nxt("model akhir (semua data)"), init=Pw if warm else None)
    prog(92, "Mengekspor ONNX…"); name = cfg.get("name") or "model_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    onnx_path, meta_path = MODELS / f"{name}.onnx", MODELS / f"{name}_metadata.json"; _export(P, chosen, mode, onnx_path)
    types = defaultdict(Counter)
    for r in recs: types[f"{r['cell'][0]},{r['cell'][1]}"][r["jenis"]] += 1
    tab = Pall.table()  # prior untuk dipakai prediksi = dari SEMUA data bersih (paling baru)
    meta = dict(name=name, parent=parent, algorithm=f"MLP 28-{chosen}-1" if chosen else "GLM (regresi L2)", target=mode, grid_km=km, features=fe.FEATURES,
                scaler={"min": mn.tolist(), "max": mx.tolist()}, metrics=metrics, opts=opts, n_orders=len(recs), n_train=n_tr, n_val=n_va, n_test=n_te,
                cleaning=dict(input=len(orders), kept=len(recs), removed=removed),
                training=dict(warm_start=warm, cv_folds=len(folds), selected=nm(chosen), candidates={nm(h): round(float(hitv[h].mean()), 4) for h in cands}, top5_se=round(se, 4),
                              days_total=n, days_train=len(folds[-1][0]), days_val=len(folds[-1][1]), days_test=len(folds[-1][2]), best_epoch=int(np.mean(beps[chosen])), epochs_run=int(ran)),
                cell_types={k: dict(v) for k, v in types.items() if k in tab["cell_rate"]}, created=dt.datetime.now().isoformat(timespec="seconds"), **tab)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), "utf-8"); _log_history(meta, days); prog(100, "Selesai")
    return dict(meta=meta, onnx_path=str(onnx_path), meta_path=str(meta_path))


def ensure_default(force=False):
    if force or not DEFAULT.exists(): train_model(fe.demo_orders(), dict(name="default_surabaya", epochs=60))
    return str(DEFAULT)


class Runtime:
    def __init__(self, onnx_path):
        import onnxruntime as ort
        p = Path(onnx_path); self.meta = json.loads(p.with_name(p.stem + "_metadata.json").read_text("utf-8"))
        if self.meta.get("features") != fe.FEATURES or "cell_slot" not in self.meta: raise ValueError("Model lama (format fitur berbeda). Latih ulang di Studio Pelatihan.")
        self.sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        self.mn = np.array(self.meta["scaler"]["min"], np.float32); self.mx = np.array(self.meta["scaler"]["max"], np.float32)

    def predict(self, cells, whens):
        """Array (len(cells), len(whens)) berisi skor potensi 0-1."""
        m = self.meta
        rows = [fe.extract(c[0], c[1], w, fe.get_priors(m, c[0], c[1], w), m["opts"]) for c in cells for w in whens]
        if not rows: return np.zeros((0, len(whens)))
        X = ((np.array(rows, np.float32) - self.mn) / (self.mx - self.mn + 1e-9)).astype(np.float32)
        o = self.sess.run(None, {"input": X})[0].ravel(); p = o if m["target"] == "prob" else 1 - np.exp(-o)
        return np.clip(p, 0, 1).reshape(len(cells), len(whens))


def latest_model():
    def ok(p):
        try: return json.loads(p.with_name(p.stem + "_metadata.json").read_text("utf-8")).get("features") == fe.FEATURES
        except Exception: return False
    c = [p for p in MODELS.glob("*.onnx") if p.stem != "default_surabaya" and ok(p)]
    return str(max(c, key=lambda p: p.stat().st_mtime)) if c else None


def quality_label(m):
    """Label berbasis BUKTI: membandingkan model dengan tebakan sederhana terkuat, memakai selang kepercayaan 90% hasil bootstrap."""
    a = m.get("roc_auc", 0)
    if "lift_ci_low" in m:
        lift, lo, hi = m["lift_vs_best_baseline"], m["lift_ci_low"], m["lift_ci_high"]
        if a < .65 or hi < 0: return "Kurang", "Model belum menemukan pola yang berguna, atau lebih buruk dari tebakan sederhana. Tambah data (≥ 14 hari) lalu latih ulang."
        if lo > 0 and lift >= .05: return "Bagus", "Terbukti lebih baik dari tebakan sederhana 'area ramai pada jam itu' (selang kepercayaan 90%, bootstrap per hari, di atas nol)."
        if lo > 0: return "Cukup", "Unggul tipis tapi nyata dari tebakan sederhana. Tambah data order agar makin akurat."
        return "Setara", "Sama baiknya dengan tebakan sederhana 'area ramai pada jam itu', belum terbukti lebih baik. Wajar bila data masih sedikit; tetap bisa dipakai. Tambah data lalu latih ulang."
    lift = m["top5_hit_rate"] - m["top5_baseline_popular"] if "top5_baseline_popular" in m else None
    if lift is not None and lift < 0.02: return "Kurang", "Model belum mengalahkan tebakan sederhana 'area paling ramai'. Tambah data lalu latih ulang."
    if a >= .8 and (lift is None or lift >= .05): return "Bagus", "Model jelas lebih baik dari tebakan sederhana."
    if a >= .7: return "Cukup", "Model menangkap pola dasar. Tambah data order agar lebih akurat."
    return "Kurang", "Model belum menemukan pola yang jelas. Tambah data (≥ 14 hari) lalu latih ulang."


def _load_weights(path):
    import onnx
    from onnx import numpy_helper as N
    w = {i.name: N.to_array(i) for i in onnx.load(str(path)).graph.initializer}
    return [w.get("W1"), w.get("b1"), w["W2"], w["b2"]]


def load_history():
    try: return json.loads((MODELS / "history.json").read_text("utf-8"))
    except Exception: return []


def _log_history(meta, days):
    """Catat tiap model yang dilatih agar pertumbuhan model dari bulan ke bulan terlihat."""
    x = meta["metrics"]; h = load_history()
    h.append(dict(name=meta["name"], created=meta["created"], n_orders=meta["n_orders"], date_from=days[0].isoformat(), date_to=days[-1].isoformat(),
                  quality=quality_label(x)[0], top5=x["top5_hit_rate"], baseline=x.get("top5_baseline_best", x["top5_baseline_popular"]), auc=x["roc_auc"],
                  warm_start=meta["training"]["warm_start"], parent=meta.get("parent")))
    (MODELS / "history.json").write_text(json.dumps(h, ensure_ascii=False, indent=1), "utf-8")
