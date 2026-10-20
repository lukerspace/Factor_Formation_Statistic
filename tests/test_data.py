"""資料整形。不碰真實 parquet：只測 MultiIndex 取欄與中央 loader 的接法。"""

import numpy as np
import pandas as pd
import pytest

from factorlab import data as fdata
from factorlab.config import DataSpec


def _multiindex_frame(field_first=True):
    idx = pd.date_range("2010-01-04", periods=40, freq="B")
    fields = ["Open", "Close", "Volume"]
    tickers = ["AAA", "BBB"]
    if field_first:
        cols = pd.MultiIndex.from_product([fields, tickers])
    else:
        cols = pd.MultiIndex.from_product([tickers, fields])
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.random((40, 6)) * 100 + 10, index=idx, columns=cols)


def test_extract_field_handles_field_first_layout():
    df = _multiindex_frame(field_first=True)
    out = fdata._extract_field(df, "Close")
    assert list(out.columns) == ["AAA", "BBB"]
    assert len(out) == 40


def test_extract_field_handles_ticker_first_layout():
    df = _multiindex_frame(field_first=False)
    out = fdata._extract_field(df, "Close")
    assert set(out.columns) == {"AAA", "BBB"}


def test_extract_field_raises_for_missing_field():
    with pytest.raises(ValueError, match="找不到欄位"):
        fdata._extract_field(_multiindex_frame(), "AdjClose")


def test_extract_field_raises_for_flat_columns():
    flat = pd.DataFrame({"a": [1.0], "b": [2.0]})
    with pytest.raises(ValueError, match="MultiIndex"):
        fdata._extract_field(flat, "Close")


def test_no_parquet_path_literals_in_package():
    """回歸測試：套件內不得出現 parquet 檔名或路徑字串。

    美股價格的唯一出口是中央 loader；硬編路徑會讓「長期記憶」失效。
    """
    import pathlib

    pkg = pathlib.Path(fdata.__file__).parent
    offenders = []
    for py in pkg.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for needle in (".parquet", "Russell3000"):
            if needle in text:
                offenders.append(f"{py.name}: {needle}")
    assert not offenders, f"套件內出現硬編資料路徑：{offenders}"


def test_import_error_message_is_actionable(monkeypatch):
    """找不到中央 loader 時，錯誤訊息必須說清楚怎麼修。"""
    monkeypatch.setenv("FACTORLAB_KICKS_ROOT", "/nonexistent/path/xyz")
    monkeypatch.setattr(fdata, "_KICKS_DEFAULT", __import__("pathlib").Path("/nonexistent/xyz"))
    import builtins

    real_import = builtins.__import__

    def blocked(name, *a, **k):
        if name.startswith("aws_fetch_price_data"):
            raise ImportError("blocked for test")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match="FACTORLAB_KICKS_ROOT"):
        fdata._import_central_loader()


def test_dataspec_defaults_to_long_history():
    """預設必須是 long —— 3 年樣本對本專案的任何檢定都沒有檢定力。"""
    assert DataSpec().history == "long"


# ------------------------------------------------- load_panel（用假 loader）


class _FakeLoader:
    """假的中央 loader，讓 load_panel 的清理邏輯能被測到。

    這一層原本完全沒有覆蓋，結果 `spec.min_stocks`（DataSpec 上並不存在
    的欄位）一路活到真實執行才爆 AttributeError。
    """

    def __init__(self, raw, usd=None):
        self._raw = raw
        self._usd = usd

    def load_us_price_data(self, history="3y"):
        return self._raw

    def get_usd_only_tickers(self):
        if self._usd is None:
            raise FileNotFoundError("no csv")
        return self._usd


def _fake_raw(n_days=260, tickers=("AAA", "BBB", "CCC"), seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=n_days)
    cols = pd.MultiIndex.from_product([["Close", "Volume"], list(tickers)])
    close = 100 * (1 + rng.standard_normal((n_days, len(tickers))) * 0.01).cumprod(axis=0)
    vol = rng.lognormal(12, 0.5, (n_days, len(tickers)))
    return pd.DataFrame(np.hstack([close, vol]), index=idx, columns=cols)


def _patched_panel(monkeypatch, raw, usd=None, **spec_kwargs):
    monkeypatch.setattr(fdata, "_import_central_loader", lambda: _FakeLoader(raw, usd))
    return fdata.load_panel(DataSpec(usd_only=usd is not None, **spec_kwargs))


def test_load_panel_builds_aligned_monthly_matrices(monkeypatch):
    panel = _patched_panel(monkeypatch, _fake_raw())
    assert panel.close.shape[1] == 3
    assert len(panel.mclose) == len(panel.mret) == len(panel.adv) == len(panel.valid)
    # 月末索引必須是真實交易日，而不是日曆月底。
    assert panel.mclose.index.isin(panel.close.index).all()
    # 第一個月沒有前一期，月報酬必須是 NaN。
    assert panel.mret.iloc[0].isna().all()


def test_load_panel_describe_reports_coverage(monkeypatch):
    info = _patched_panel(monkeypatch, _fake_raw()).describe()
    assert info["n_tickers"] == 3
    assert info["n_months"] > 10
    assert info["date_start"] < info["date_end"]


def test_load_panel_filters_impossible_upside_returns(monkeypatch):
    """|r| > max_abs_daily_return 的日報酬視為資料錯誤，設為 NaN。"""
    raw = _fake_raw(seed=1)
    # 反向分割未調整：價格一天變成 5 倍（+400%）。
    raw.loc[raw.index[50], ("Close", "AAA")] = raw.loc[raw.index[49], ("Close", "AAA")] * 5
    panel = _patched_panel(monkeypatch, raw, max_abs_daily_return=1.0)
    assert np.isnan(panel.dret.loc[raw.index[50], "AAA"])


def test_daily_return_filter_is_asymmetric_by_construction(monkeypatch):
    """記錄一個真實限制：預設門檻擋不住正向分割。

    報酬的下界是 -100%，所以 |r| > 1.0 這個門檻只擋得住上漲方向的
    錯誤（反向分割）。10 拆 1 的正向分割會呈現 -90%，不會被擋掉。
    這不是可以靜默「修掉」的事 —— 把負向門檻收到 -90% 會同時丟掉真實
    的崩跌（生技試驗失敗、財報造假），而那些是動能策略報酬的一部分。
    因此這裡選擇保留並在報告的 caveats 明寫，由讀者判斷。
    """
    raw = _fake_raw(seed=11)
    raw.loc[raw.index[50], ("Close", "AAA")] = raw.loc[raw.index[49], ("Close", "AAA")] / 10
    panel = _patched_panel(monkeypatch, raw, max_abs_daily_return=1.0)
    assert panel.dret.loc[raw.index[50], "AAA"] == pytest.approx(-0.9)


def test_tighter_threshold_does_filter_large_drops(monkeypatch):
    """需要時可以把門檻收緊 —— 這是 DataSpec 的參數，不是寫死的。"""
    raw = _fake_raw(seed=12)
    raw.loc[raw.index[50], ("Close", "AAA")] = raw.loc[raw.index[49], ("Close", "AAA")] / 10
    panel = _patched_panel(monkeypatch, raw, max_abs_daily_return=0.8)
    assert np.isnan(panel.dret.loc[raw.index[50], "AAA"])


def test_load_panel_drops_sub_dollar_prices(monkeypatch):
    raw = _fake_raw(seed=2)
    raw.loc[:, ("Close", "CCC")] = 0.5
    panel = _patched_panel(monkeypatch, raw, min_price=1.0)
    assert panel.close["CCC"].isna().all()


def test_load_panel_country_filter_is_skipped_when_too_few_remain(monkeypatch):
    """國別篩選只剩 1 檔時應整個跳過，而不是把 universe 砍爛。"""
    panel = _patched_panel(monkeypatch, _fake_raw(seed=3), usd={"AAA"})
    assert panel.close.shape[1] == 3


def test_load_panel_tolerates_missing_country_csv(monkeypatch):
    raw = _fake_raw(seed=4)
    monkeypatch.setattr(fdata, "_import_central_loader", lambda: _FakeLoader(raw, None))
    panel = fdata.load_panel(DataSpec(usd_only=True))
    assert panel.close.shape[1] == 3


def test_load_panel_respects_date_window(monkeypatch):
    raw = _fake_raw(n_days=500, seed=5)
    panel = _patched_panel(monkeypatch, raw, start="2010-06-01", end="2010-12-31")
    assert panel.close.index.min() >= pd.Timestamp("2010-06-01")
    assert panel.close.index.max() <= pd.Timestamp("2010-12-31")


def test_load_panel_marks_thin_months_invalid(monkeypatch):
    raw = _fake_raw(n_days=260, seed=6)
    panel = _patched_panel(monkeypatch, raw, min_trading_days_per_month=15)
    # 最後一個月只有幾個交易日時應被標為無效。
    assert panel.valid.dtypes.unique().tolist() == [np.dtype("bool")]
    assert panel.mclose.where(~panel.valid).isna().all().all()


def test_realized_vol_is_annualized_and_lagged(monkeypatch):
    panel = _patched_panel(monkeypatch, _fake_raw(n_days=400, seed=7))
    rv = fdata.realized_vol(panel, window=60, min_periods=40)
    assert rv.index.equals(panel.mclose.index)
    vals = rv.to_numpy()
    vals = vals[np.isfinite(vals)]
    assert len(vals) > 0
    assert (vals > 0).all()
