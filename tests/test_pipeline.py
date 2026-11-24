"""Tests sur données SYNTHÉTIQUES (jamais utilisées pour les résultats du projet)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import forecast as f  # noqa: E402


def _wb_csv(path, years, values, code="XXX"):
    header = '"Data Source","WDI"\n\n"Last Updated Date","test"\n\n'
    cols = ["Country Name", "Country Code", "Indicator Name", "Indicator Code"] + [str(y) for y in years]
    row = ["Pays test", code, "Air transport, passengers carried", "IS.AIR.PSGR"] + list(values)
    path.write_text(header + ",".join(f'"{c}"' for c in cols) + "\n" + ",".join(f'"{v}"' for v in row) + "\n")


def _series():
    years = np.arange(1990, 2023)
    rng = np.random.default_rng(0)
    v = 1e5 * np.exp(0.06 * (years - 1990) + rng.normal(0, 0.03, len(years)))
    v[years == 2020] *= 0.4
    v[years == 2021] *= 0.6
    return years, v


def test_load_wb_format(tmp_path):
    years, v = _series()
    p = tmp_path / "wb.csv"
    _wb_csv(p, years, v)
    s = f.load_series(p, "XXX")
    assert len(s) == len(years) and s.index.min() == 1990


def test_load_two_column(tmp_path):
    years, v = _series()
    p = tmp_path / "s.csv"
    pd.DataFrame({"year": years, "value": v}).to_csv(p, index=False)
    assert len(f.load_series(p)) == len(years)


def test_eval_and_scenario():
    years, v = _series()
    s = pd.Series(v, index=years)
    pre = s[s.index <= 2019]
    res = f.summarize(f.rolling_origin_eval(pre, 15))
    assert set(res.modele) == set(f.MODELS)
    best = res[res.horizon == 1].iloc[0].modele
    scen = f.trend_scenario(s, best, 2019, 2025)
    # le choc COVID doit apparaître comme un écart négatif en 2020
    assert scen.loc[scen.annee == 2020, "ecart_pct"].iloc[0] < -30


def test_longest_contiguous():
    s = pd.Series([1.0, 2, 3, 5, 6, 7, 8, 9], index=[1990, 1991, 1992, 1995, 1996, 1997, 1998, 1999])
    out = f.longest_contiguous(s)
    assert out.index.min() == 1995 and len(out) == 5


def test_prepare_fills_short_gaps_only():
    yrs = list(range(1990, 2000)) + [2003] + list(range(2004, 2012))  # trou de 3 ans (2000-2002)
    s = pd.Series(np.arange(1, len(yrs) + 1) * 1000.0, index=yrs)
    s2, filled = f.prepare(s, 1990, max_gap=2)
    assert s2.index.min() == 2003 or s2.index.min() == 1990  # bloc continu le plus long
    s.loc[2003] = np.nan
    yrs2 = [y for y in range(1990, 2012) if y not in (1995, 1996)]
    s3 = pd.Series(1000.0 * np.arange(1, len(yrs2) + 1), index=yrs2)
    s4, filled4 = f.prepare(s3, 1990, max_gap=2)
    assert filled4 == {1995, 1996} and len(s4) == 22


def test_detect_breaks():
    s = pd.Series([100.0, 110, 120, 400, 420, 430], index=[2000, 2001, 2002, 2003, 2004, 2005])
    assert [y for y, _ in f.detect_breaks(s)] == [2003]
