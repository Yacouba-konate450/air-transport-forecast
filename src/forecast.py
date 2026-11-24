"""Prévision du trafic aérien passagers (indicateur Banque mondiale IS.AIR.PSGR).

Usage :
    python src/forecast.py --csv data/API_IS.AIR.PSGR_DS2_en_csv_v2_XXXX.csv --country MLI

Étapes :
1. Chargement de la série annuelle (format CSV Banque mondiale ou 2 colonnes year,value).
2. Évaluation hors échantillon (origine glissante) de 4 modèles sur la période AVANT COVID.
3. Scénario tendanciel : le meilleur modèle, ajusté jusqu'à l'année de coupure,
   prolonge la série ; l'écart avec les valeurs observées mesure le choc COVID-19.
"""
from __future__ import annotations

import argparse
import warnings
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

MODELS = ["naif", "derive", "ets_amorti", "arima"]
HORIZONS = (1, 3)


# --------------------------------------------------------------------------- données
def load_series(path: str | Path, country: str = "MLI") -> pd.Series:
    """Retourne une série annuelle indexée par année (int), valeurs > 0, sans NaN."""
    path = Path(path)
    head = path.read_text(encoding="utf-8-sig", errors="ignore")[:200]
    if "Data Source" in head:  # export standard Banque mondiale (4 lignes d'en-tête)
        df = pd.read_csv(path, skiprows=4)
        row = df[df["Country Code"] == country]
        if row.empty:
            raise ValueError(f"Pays {country} introuvable dans {path.name}")
        years = [c for c in df.columns if c.isdigit()]
        s = row[years].iloc[0].astype(float)
        s.index = s.index.astype(int)
    else:  # format simple : year,value
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        s = df.set_index(df.columns[0])[df.columns[1]].astype(float)
        s.index = s.index.astype(int)
    s = s.dropna()
    s = s[s > 0].sort_index()
    if s.empty:
        raise ValueError("Série vide après nettoyage")
    return s


# --------------------------------------------------------------------------- modèles
def _fit_arima(ylog: np.ndarray):
    """Sélection par AIC de (p,1,q) avec dérive, p,q dans 0..2."""
    best, best_aic = None, np.inf
    for p, q in product(range(3), range(3)):
        try:
            res = SARIMAX(ylog, order=(p, 1, q), trend="t").fit(disp=False)
        except Exception:
            continue
        if np.isfinite(res.aic) and res.aic < best_aic:
            best, best_aic = res, res.aic
    if best is None:
        raise RuntimeError("Aucun ARIMA n'a convergé")
    return best


def forecast(name: str, train: pd.Series, h: int) -> np.ndarray:
    """Prévision à 1..h pas (en niveau). Modèles estimés sur le log de la série."""
    y = train.to_numpy(dtype=float)
    ylog = np.log(y)
    if name == "naif":
        return np.repeat(y[-1], h)
    if name == "derive":
        slope = (ylog[-1] - ylog[0]) / (len(ylog) - 1)
        return np.exp(ylog[-1] + slope * np.arange(1, h + 1))
    if name == "ets_amorti":
        res = ExponentialSmoothing(ylog, trend="add", damped_trend=True).fit()
        return np.exp(res.forecast(h))
    if name == "arima":
        return np.exp(_fit_arima(ylog).forecast(h))
    raise ValueError(name)


# --------------------------------------------------------------------------- évaluation
def rolling_origin_eval(s: pd.Series, min_train: int = 15, skip=()) -> pd.DataFrame:
    """Origine glissante (fenêtre croissante). Une ligne par modèle, horizon, origine."""
    rows = []
    n = len(s)
    for t in range(min_train, n):
        train = s.iloc[:t]
        for m in MODELS:
            fc = forecast(m, train, max(HORIZONS))
            for h in HORIZONS:
                if t + h - 1 < n and int(s.index[t + h - 1]) not in skip:
                    actual = s.iloc[t + h - 1]
                    rows.append(
                        dict(modele=m, horizon=h, annee_cible=int(s.index[t + h - 1]),
                             prevu=fc[h - 1], observe=actual, erreur=fc[h - 1] - actual)
                    )
    return pd.DataFrame(rows)


def summarize(ev: pd.DataFrame) -> pd.DataFrame:
    g = ev.assign(ae=ev.erreur.abs(), se=ev.erreur**2, ape=(ev.erreur / ev.observe).abs() * 100)
    out = g.groupby(["horizon", "modele"]).agg(
        n=("ae", "size"), MAE=("ae", "mean"), RMSE=("se", lambda x: np.sqrt(x.mean())),
        MAPE=("ape", "mean"), biais=("erreur", "mean"),
    )
    return out.reset_index().sort_values(["horizon", "RMSE"])


# --------------------------------------------------------------------------- scénario
def trend_scenario(s: pd.Series, best: str, cutoff: int, end_year: int) -> pd.DataFrame:
    train = s[s.index <= cutoff]
    years = np.arange(cutoff + 1, end_year + 1)
    fc = forecast(best, train, len(years))
    df = pd.DataFrame({"annee": years, "tendanciel": fc})
    df["observe"] = s.reindex(years).to_numpy()
    df["ecart_pct"] = (df.observe / df.tendanciel - 1) * 100
    return df


def plot(s, scen, best, cutoff, country, path):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(s.index, s / 1e6, "o-", color="#1f3a5f", lw=1.8, ms=3.5, label="Observé")
    ax.plot(np.r_[cutoff, scen.annee], np.r_[s.get(cutoff, np.nan), scen.tendanciel] / 1e6,
            "--", color="#c0562b", lw=1.8, label=f"Scénario tendanciel ({best}, ajusté ≤ {cutoff})")
    ax.axvspan(2020 - 0.5, 2021 + 0.5, color="grey", alpha=0.15, label="COVID-19")
    ax.set_title(f"Passagers aériens transportés – {country}")
    ax.set_ylabel("Millions de passagers")
    ax.set_xlabel("Année")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def prepare(s: pd.Series, start: int, max_gap: int = 2):
    """Coupe avant `start`, comble par interpolation log-linéaire les trous de <= max_gap ans,
    puis garde le plus long bloc continu. Retourne (série, années interpolées)."""
    s = s[s.index >= start]
    full = np.log(s.reindex(range(int(s.index.min()), int(s.index.max()) + 1)))
    isna = full.isna()
    run = isna.ne(isna.shift()).cumsum()
    runlen = isna.groupby(run).transform("sum")
    fill = isna & (runlen <= max_gap)
    interp = full.interpolate(limit_area="inside")
    full[fill] = interp[fill]
    filled = {int(y) for y in full.index[fill]}
    out = np.exp(full.dropna())
    return longest_contiguous(out), filled


def longest_contiguous(s: pd.Series) -> pd.Series:
    """Plus long bloc d'années consécutives (les trous fausseraient les modèles)."""
    grp = (s.index.to_series().diff() != 1).cumsum()
    sizes = grp.value_counts()
    return s[grp.to_numpy() == sizes.idxmax()]


def detect_breaks(s: pd.Series, thr: float = 0.7) -> list[tuple[int, float]]:
    """Années où le trafic est multiplié ou divisé par plus de e^thr (~2) d'une année à l'autre."""
    s = s[s.index.to_series().diff().reindex(s.index).fillna(1) == 1]  # années consécutives seulement
    d = np.log(s).diff().dropna()
    return [(int(y), float(np.exp(v))) for y, v in d[d.abs() > thr].items()]


def plot_raw(full: pd.Series, code: str, path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(full.index, full / 1e6, "o-", ms=3, color="#1f3a5f")
    ax.set_title(f"Série brute – {code} (Banque mondiale, IS.AIR.PSGR)")
    ax.set_ylabel("Millions de passagers")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_country(csv, code, a, out_root):
    out = out_root / code
    out.mkdir(parents=True, exist_ok=True)
    full = load_series(csv, code)
    plot_raw(full, code, out / "serie_brute.png")
    brks = detect_breaks(full[full.index >= a.start])
    hors_covid = [b for b in brks if b[0] not in (2020, 2021)]
    s, filled = prepare(full, a.start, a.max_gap)
    print(f"\n===== {code} : brut {full.index.min()}-{full.index.max()} ({len(full)} points) ; "
          f"retenu {s.index.min()}-{s.index.max()} ({len(s)} points, dont {len(filled & set(s.index))} interpolés)")
    if brks:
        print("  Ruptures (facteur d'une année à l'autre) : "
              + ", ".join(f"{y} x{f:.2f}" for y, f in brks))
    pre = s[s.index <= a.cutoff]
    if len(pre) <= a.min_train + 3:
        print(f"  Ignoré : seulement {len(pre)} points avant {a.cutoff + 1}")
        return dict(pays=code, debut=int(s.index.min()), fin=int(s.index.max()), points=len(s),
                    ruptures=len(brks), ruptures_hors_covid=len(hors_covid),
                    statut=f"ignoré ({len(pre)} points continus)")
    ev = rolling_origin_eval(pre, a.min_train, filled)
    res = summarize(ev)
    ev.to_csv(out / "evaluation_detail.csv", index=False)
    res.to_csv(out / "evaluation_resume.csv", index=False)
    print(res.round(2).to_string(index=False))
    h1 = res[res.horizon == 1]
    best = h1.iloc[0].modele
    best_trend = h1[h1.modele != "naif"].iloc[0].modele  # le naïf n'a pas de tendance
    rmse = h1.set_index("modele").RMSE
    print(f"  Meilleur modèle (RMSE, h=1) : {best} ; meilleur modèle à tendance : {best_trend}")
    row = dict(pays=code, debut=int(s.index.min()), fin=int(s.index.max()), points=len(s),
               meilleur=best, MAPE_h1=h1[h1.modele == best].MAPE.iloc[0],
               RMSE_tendance_vs_naif=rmse[best_trend] / rmse["naif"], tendance=best_trend,
               ruptures=len(brks), ruptures_hors_covid=len(hors_covid), statut="évalué")
    row["fiabilite"] = "correcte" if (len(hors_covid) <= 1 and row["MAPE_h1"] < 30) else "à interpréter avec réserve"
    if s.index.max() > a.cutoff:
        scen = trend_scenario(s, best_trend, a.cutoff, int(s.index.max()) + 3)
        scen.to_csv(out / "scenario_tendanciel.csv", index=False)
        print(scen.round(1).to_string(index=False))
        plot(s, scen, best_trend, a.cutoff, code, out / "trafic_vs_tendance.png")
    else:
        print(f"  Série terminée en {s.index.max()} : pas de scénario COVID pour ce pays")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--country", default="MLI", help="un ou plusieurs codes ISO3 séparés par des virgules")
    ap.add_argument("--cutoff", type=int, default=2019, help="dernière année avant COVID")
    ap.add_argument("--min-train", type=int, default=15)
    ap.add_argument("--start", type=int, default=1990, help="première année utilisée")
    ap.add_argument("--max-gap", type=int, default=2, help="trous comblés par interpolation (années)")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    out_root = Path(a.out)
    rows = []
    for code in [c.strip().upper() for c in a.country.split(",")]:
        try:
            r = run_country(a.csv, code, a, out_root)
        except ValueError as e:
            print(f"\n===== {code} : {e}")
            continue
        if r:
            rows.append(r)
    if rows:
        summ = pd.DataFrame(rows)
        out_root.mkdir(parents=True, exist_ok=True)
        summ.to_csv(out_root / "synthese_pays.csv", index=False)
        print("\nSynthèse :\n", summ.round(1).to_string(index=False))
    print(f"\nRésultats écrits dans {out_root}/")


if __name__ == "__main__":
    main()
