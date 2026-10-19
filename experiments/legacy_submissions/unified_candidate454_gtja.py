"""Frozen readable GTJA191 Polars runtime; generated from vendored MIT source."""
from __future__ import annotations


# ---- third_party/aurumq_gtja191/aurumq_rl/factors/registry.py ----
"""Factor registry for AurumQ alpha101 + gtja191 unified factor library.

Single source of truth for factor metadata + polars callable implementation.
Used by:

* ``aurumq.factors.alpha101`` / ``aurumq.factors.gtja191`` (via symlink) for
  panel computation pipelines.
* ``aurumq.rules.aqml_polars_compiler`` for the ``resolve_for_aqml`` hook —
  when a user strategy expression like ``Rank(close) - alpha001`` references
  a factor symbol, the compiler resolves it via this registry instead of
  treating it as an ordinary panel column.
* ``aurumq_rl.factors._docs`` to extract docstring → markdown documentation.

Design notes
------------
* ``FactorEntry`` is frozen so registries cannot accidentally mutate metadata.
* ``impl`` is the canonical polars implementation. Each factor function takes
  a single ``pl.DataFrame`` (the enriched panel) and returns a ``pl.Series``
  aligned to the panel rows.
* ``legacy_aqml_expr`` is preserved on alpha101 entries that were migrated
  from the legacy ``aqml_strategy`` string-expression library; it is used as
  a numerical cross-check during the migration period and may be removed
  once parity is verified.
* ``quality_flag``: ``0`` = ok, ``1`` = errata-conservative (gtja191
  ambiguous formulas), ``2`` = stub.
"""
import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    import polars as pl
    FactorImpl = Callable[['pl.DataFrame'], 'pl.Series']
else:
    FactorImpl = Callable
__all__ = ['FactorEntry', 'ALPHA101_REGISTRY', 'GTJA191_REGISTRY', 'FACTOR_CLIP_LIMIT', 'list_all_factors', 'register_alpha101', 'register_gtja191', 'resolve_for_aqml', 'sanitize_factor_series']
FACTOR_CLIP_LIMIT: float = 1000000.0

def sanitize_factor_series(series: pl.Series) -> pl.Series:
    """Replace ±inf with null and clip finite values to ±``FACTOR_CLIP_LIMIT``.

    Defends downstream consumers (``np.percentile``, cross-section z-score,
    ML training) against factor-library overflow / divide-by-zero artifacts:
        - ±inf → null (numpy's percentile returns nan + warns on inf input)
        - |x| > 1e6 → clipped (gtja_017's ``rank ^ delta`` blew up to 1e+302)
        - finite, in-range → unchanged

    Returns a new Float64 Series of the same length and name.
    Cheap (one with_columns expression), so safe to apply on every factor.
    """
    import polars as pl
    if series.dtype != pl.Float64:
        series = series.cast(pl.Float64, strict=False)
    name = series.name
    return pl.DataFrame({name: series}).with_columns(pl.when(pl.col(name).is_infinite()).then(None).otherwise(pl.col(name).clip(-FACTOR_CLIP_LIMIT, FACTOR_CLIP_LIMIT)).alias(name))[name]

def _wrap_impl_with_sanitizer(impl: FactorImpl) -> FactorImpl:
    """Wrap a factor ``impl(df) -> pl.Series`` so its output is sanitized."""

    def _sanitized(df):
        return sanitize_factor_series(impl(df))
    _sanitized.__name__ = getattr(impl, '__name__', '_sanitized_impl')
    _sanitized.__doc__ = getattr(impl, '__doc__', None)
    _sanitized.__wrapped__ = impl
    return _sanitized

@dataclass(frozen=True)
class FactorEntry:
    """Metadata + callable for a single factor."""
    id: str
    impl: FactorImpl
    direction: Literal['normal', 'reverse']
    category: str
    description: str
    legacy_aqml_expr: str | None = None
    quality_flag: int = 0
    references: tuple[str, ...] = field(default_factory=tuple)
    formula_doc_path: str = ''
ALPHA101_REGISTRY: dict[str, FactorEntry] = {}
GTJA191_REGISTRY: dict[str, FactorEntry] = {}

def register_alpha101(entry: FactorEntry) -> FactorEntry:
    """Register a factor in the alpha101 registry (idempotent on identical entry).

    The entry's ``impl`` is automatically wrapped with :func:`sanitize_factor_series`
    so every registered factor produces inf-free, clipped output regardless of how
    the underlying formula handles divide-by-zero / overflow.
    """
    sanitized_entry = dataclasses.replace(entry, impl=_wrap_impl_with_sanitizer(entry.impl))
    if entry.id in ALPHA101_REGISTRY and ALPHA101_REGISTRY[entry.id] is not sanitized_entry:
        existing_impl = getattr(ALPHA101_REGISTRY[entry.id].impl, '__wrapped__', ALPHA101_REGISTRY[entry.id].impl)
        if existing_impl is not entry.impl:
            raise ValueError(f'alpha101 factor {entry.id!r} already registered with a different entry')
    ALPHA101_REGISTRY[entry.id] = sanitized_entry
    return sanitized_entry

def register_gtja191(entry: FactorEntry) -> FactorEntry:
    """Register a factor in the gtja191 registry (idempotent on identical entry).

    See :func:`register_alpha101` for the auto-sanitization contract.
    """
    sanitized_entry = dataclasses.replace(entry, impl=_wrap_impl_with_sanitizer(entry.impl))
    if entry.id in GTJA191_REGISTRY and GTJA191_REGISTRY[entry.id] is not sanitized_entry:
        existing_impl = getattr(GTJA191_REGISTRY[entry.id].impl, '__wrapped__', GTJA191_REGISTRY[entry.id].impl)
        if existing_impl is not entry.impl:
            raise ValueError(f'gtja191 factor {entry.id!r} already registered with a different entry')
    GTJA191_REGISTRY[entry.id] = sanitized_entry
    return sanitized_entry

def list_all_factors() -> dict[str, FactorEntry]:
    """Return a merged view of alpha101 + gtja191 registries.

    Mutating the returned dict does NOT affect the underlying registries.
    Both factor families share an id-namespace by convention (``alpha`` and
    ``gtja_`` prefixes); collisions raise.
    """
    overlap = ALPHA101_REGISTRY.keys() & GTJA191_REGISTRY.keys()
    if overlap:
        raise RuntimeError(f'Factor id collision across registries: {sorted(overlap)}')
    return {**ALPHA101_REGISTRY, **GTJA191_REGISTRY}

def resolve_for_aqml(name: str, df: pl.DataFrame) -> pl.Series:
    """Hook called by ``aqml_polars_compiler`` to resolve a factor symbol.

    Parameters
    ----------
    name :
        Factor id, e.g. ``"alpha001"`` or ``"gtja_042"``.
    df :
        The enriched panel DataFrame the AQML compiler is currently
        evaluating against.

    Returns
    -------
    pl.Series
        Output of the registered factor implementation.

    Raises
    ------
    KeyError
        If ``name`` is not in any registry. The caller (AQML compiler) should
        fall back to treating the symbol as an ordinary panel column.
    """
    factors = list_all_factors()
    if name not in factors:
        raise KeyError(name)
    entry = factors[name]
    return entry.impl(df)

# ---- third_party/aurumq_gtja191/aurumq_rl/factors/alpha101/_ops.py ----
"""Shared polars operators for alpha101 factor implementations.

All time-series operators partition by ``stock_code`` (assume the panel
is sorted by ``[stock_code, trade_date]`` ascending). Cross-sectional
operators partition by ``trade_date``.

Operators are pure :class:`polars.Expr` builders — they do not eagerly
evaluate. Compose with ``df.with_columns(...)`` / ``df.select(...)``.

Conventions
-----------
* ``window`` arguments are inclusive of the current row (length-N window
  consumes N rows).
* Rolling outputs are NaN/null for the first ``window-1`` rows of each
  partition (matches pandas ``rolling(min_periods=window)`` semantics).
* All numeric outputs are cast to / preserved as :class:`pl.Float64`.

Tie-breaking & STHSF divergence
-------------------------------
``ts_argmax`` / ``ts_argmin`` use polars ``Series.arg_max`` /
``Series.arg_min`` semantics, which return the **FIRST** occurrence of
the extremum. Pandas' ``rolling.apply(np.argmax)`` returns the **LAST**
occurrence on ties. The locked STHSF reference parquet was produced with
pandas, so a small fraction of windows containing exact ties will diverge
from the reference. In synthetic GBM data ties are rare (<0.1% of
windows), so the suite-wide ``rtol=1e-3, atol=1e-3`` tolerance absorbs
the divergence.
"""
import polars as pl
TS_PART = 'stock_code'
'Partition column for time-series operators (rolling within stock).'
CS_PART = 'trade_date'
'Partition column for cross-sectional operators (rank within day).'
EPS_DIV: float = 1e-09
'Default denominator floor for safe_div: |den| < this → null.'
EPS_LOG: float = 1e-12
'Default floor for safe_log: input < this is clipped up before taking log.'
SAFE_POW_EXP_LO: float = -3.0
SAFE_POW_EXP_HI: float = 3.0
'Default exponent clip for safe_pow_clip: bounds growth/decay rate.'

def safe_div(num: pl.Expr, den: pl.Expr, eps: float=EPS_DIV) -> pl.Expr:
    """Division that returns NULL when ``|den| < eps`` instead of inf/nan.

    Used wherever the denominator can legitimately be ~0 on real A-share
    data (limit-up days, suspended stocks, zero-volume bars). Compared to
    ``num / (den + eps)`` this preserves sign and produces a clean missing
    indicator instead of a saturated value.
    """
    return pl.when(den.abs() > eps).then(num / den).otherwise(None)

def safe_log(x: pl.Expr, eps: float=EPS_LOG) -> pl.Expr:
    """``log`` with a positive floor so log(0) does not return -inf.

    For non-positive inputs returns log(eps) (a large negative number);
    callers that prefer NULL can chain ``.fill_nan(None)``.
    """
    return x.clip(lower_bound=eps).log()

def safe_pow_clip(base: pl.Expr, exponent: pl.Expr, exp_lo: float=SAFE_POW_EXP_LO, exp_hi: float=SAFE_POW_EXP_HI) -> pl.Expr:
    """Power with bounded exponent so ``rank ^ delta`` cannot overflow.

    A-share data sometimes produces ``delta(close, 5)`` values in the
    range [-50, +50] when adj_factor is glitchy (see audit doc). Combined
    with ``rank`` ∈ [0, 1] this gives ``0^-50 = inf`` or ``rank^50 ≈ 0``
    that float32 cast turns into 0/inf. Clipping the exponent to ±3 keeps
    economically meaningful momentum (3-fold growth/decay max) while
    eliminating overflow.
    """
    return base ** exponent.clip(exp_lo, exp_hi)

def ts_mean(col: pl.Expr, window: int) -> pl.Expr:
    """Rolling arithmetic mean over ``window`` rows, per stock."""
    return col.rolling_mean(window_size=window, min_samples=window).over(TS_PART)

def ts_sum(col: pl.Expr, window: int) -> pl.Expr:
    """Rolling sum over ``window`` rows, per stock."""
    return col.rolling_sum(window_size=window, min_samples=window).over(TS_PART)

def ts_std(col: pl.Expr, window: int) -> pl.Expr:
    """Rolling sample stddev (ddof=1) over ``window`` rows, per stock."""
    return col.rolling_std(window_size=window, min_samples=window).over(TS_PART)

def ts_min(col: pl.Expr, window: int) -> pl.Expr:
    """Rolling minimum over ``window`` rows, per stock."""
    return col.rolling_min(window_size=window, min_samples=window).over(TS_PART)

def ts_max(col: pl.Expr, window: int) -> pl.Expr:
    """Rolling maximum over ``window`` rows, per stock."""
    return col.rolling_max(window_size=window, min_samples=window).over(TS_PART)

def ts_median(col: pl.Expr, window: int) -> pl.Expr:
    """Rolling median over ``window`` rows, per stock."""
    return col.rolling_median(window_size=window, min_samples=window).over(TS_PART)

def ts_skew(col: pl.Expr, window: int) -> pl.Expr:
    """Rolling skewness over ``window`` rows, per stock.

    Uses polars' native ``rolling_skew`` (Fisher-Pearson moment-based).
    """
    return col.rolling_skew(window_size=window, min_samples=window).over(TS_PART)

def ts_kurt(col: pl.Expr, window: int) -> pl.Expr:
    """Rolling Fisher kurtosis (excess) over ``window`` rows, per stock."""
    return col.rolling_kurtosis(window_size=window, min_samples=window).over(TS_PART)

def ts_zscore(col: pl.Expr, window: int) -> pl.Expr:
    """Rolling z-score: ``(x - mean) / std`` over ``window`` rows, per stock."""
    mean = col.rolling_mean(window_size=window, min_samples=window)
    std = col.rolling_std(window_size=window, min_samples=window)
    return ((col - mean) / std).over(TS_PART)

def _window_values(col: pl.Expr, window: int) -> list[pl.Expr]:
    """Return oldest->newest values in the current rolling window."""
    return [col.shift(window - 1 - i).over(TS_PART) for i in range(window)]

def _window_valid(values: list[pl.Expr]) -> pl.Expr:
    """True when a shifted rolling window contains no null or NaN values."""
    return pl.all_horizontal([value.is_not_null() & ~value.is_nan() for value in values])

def _window_arg_extreme(col: pl.Expr, window: int, *, kind: str, tie: str) -> pl.Expr:
    """Vectorized rolling argmax/argmin over a fixed-size per-stock window."""
    values = _window_values(col, window)
    valid = _window_valid(values)
    extreme = pl.max_horizontal(values) if kind == 'max' else pl.min_horizontal(values)
    order = range(window - 1, -1, -1) if tie == 'first' else range(window)
    out = pl.lit(None, dtype=pl.Float64)
    for idx in order:
        out = pl.when(values[idx] == extreme).then(float(idx)).otherwise(out)
    return pl.when(valid).then(out).otherwise(None).cast(pl.Float64)

def ts_argmax(col: pl.Expr, window: int) -> pl.Expr:
    """Position (0..window-1) of max within last ``window`` rows, per stock.

    Returned as :class:`pl.Float64` for downstream rank/arithmetic safety.
    Windows with any null/NaN return null (pandas-rolling parity).

    .. note::
       **Tie convention**: returns the index of the **first** occurrence of
       the max (polars ``arg_max`` semantics). Pandas/STHSF returns the
       **last** occurrence — see module docstring.
    """
    return _window_arg_extreme(col, window, kind='max', tie='first')

def ts_argmin(col: pl.Expr, window: int) -> pl.Expr:
    """Position (0..window-1) of min within last ``window`` rows, per stock.

    Same tie convention as :func:`ts_argmax` — first occurrence wins.
    """
    return _window_arg_extreme(col, window, kind='min', tie='first')

def ts_argmax_last(col: pl.Expr, window: int) -> pl.Expr:
    """``ts_argmax`` variant where ties resolve to the **last** occurrence.

    Used by alpha factors whose authors selected the most-recent peak when
    the window contains repeated maxima. Returned as :class:`pl.Float64`.
    """
    return _window_arg_extreme(col, window, kind='max', tie='last')

def ts_argmin_last(col: pl.Expr, window: int) -> pl.Expr:
    """``ts_argmin`` variant where ties resolve to the **last** occurrence.

    Companion of :func:`ts_argmax_last`. See its docstring.
    """
    return _window_arg_extreme(col, window, kind='min', tie='last')

def ts_rank(col: pl.Expr, window: int) -> pl.Expr:
    """Rolling rank of the **last** value within the past ``window`` rows.

    Output is normalised into ``[0, 1]`` using ``(rank - 1) / (n - 1)``,
    matching WorldQuant 101 ``ts_rank`` convention. ``method='average'``
    is used for tie-breaking. Returns null until the window is full or if
    any value in the window is null.

    Implementation prefers the native :func:`pl.Expr.rolling_rank`
    (polars 1.x stable) and divides by ``window - 1`` for the [0, 1] scale.
    """
    if window < 2:
        return pl.lit(0.0, dtype=pl.Float64)
    raw_rank = col.rolling_rank(window_size=window, method='average', min_samples=window)
    return ((raw_rank - 1.0) / float(window - 1)).over(TS_PART).cast(pl.Float64)

def ts_rank_int(col: pl.Expr, window: int) -> pl.Expr:
    """Rolling **integer** rank (1..window) of the last value, per stock.

    Differs from :func:`ts_rank` in that the output is the raw average rank
    (1..window) rather than the normalised pct rank in ``[0, 1]``. Provided
    for STHSF parity (their helper returns 1..window) and for alpha factors
    that arithmetic-combine the rank with non-rank quantities.

    Tie semantics match pandas/scipy ``rankdata``:
    ``avg_rank = lt + (eq + 1) / 2`` — equivalent to ``method='average'``.
    """
    values = _window_values(col, window)
    valid = _window_valid(values)
    last = values[-1]
    eq = pl.sum_horizontal([(value == last).cast(pl.Float64) for value in values])
    lt = pl.sum_horizontal([(value < last).cast(pl.Float64) for value in values])
    rank = lt + (eq + 1.0) / 2.0
    return pl.when(valid).then(rank).otherwise(None).cast(pl.Float64)

def ts_decay_linear(col: pl.Expr, window: int) -> pl.Expr:
    """Linearly weighted moving average with weights ``[w, w-1, …, 1]/sum``.

    The most recent observation gets weight ``window``; the oldest gets
    weight ``1``. Output is null for the first ``window - 1`` rows of
    each stock partition.

    Built as a sum of weighted ``shift(i)`` exprs — stays vectorised and
    avoids the ``rolling_map`` Python callback.
    """
    if window < 1:
        raise ValueError(f'window={window} must be >= 1')
    weights = [window - i for i in range(window)]
    total = float(sum(weights))
    shifted = [col.shift(i).over(TS_PART) * (w / total) for i, w in enumerate(weights)]
    out = shifted[0]
    for term in shifted[1:]:
        out = out + term
    return out.cast(pl.Float64)

def wma(col: pl.Expr, window: int) -> pl.Expr:
    """Linear weighted MA — alias of :func:`ts_decay_linear`.

    Provided for STHSF compatibility (their helper is named ``wma``).
    """
    return ts_decay_linear(col, window)

def sma(col: pl.Expr, window: int) -> pl.Expr:
    """Simple moving average — alias of :func:`ts_mean`.

    Provided for STHSF compatibility (their helper is named ``sma``).
    """
    return ts_mean(col, window)

def ts_product(col: pl.Expr, window: int) -> pl.Expr:
    """Rolling product over ``window`` rows, per stock.

    Implemented as ``exp(rolling_sum(log(x)))``. **Caveat**: undefined for
    non-positive inputs — caller is responsible for ensuring positivity.
    """
    log_x = col.log()
    return log_x.rolling_sum(window_size=window, min_samples=window).over(TS_PART).exp()

def ts_corr(x: pl.Expr, y: pl.Expr, window: int) -> pl.Expr:
    """Rolling Pearson correlation between two columns, per stock.

    Uses :func:`polars.rolling_corr` — vectorised and fast.
    """
    return pl.rolling_corr(x, y, window_size=window, min_samples=window).over(TS_PART)

def ts_corr_safe(x: pl.Expr, y: pl.Expr, window: int) -> pl.Expr:
    """Same as :func:`ts_corr` but maps NaN outputs to null.

    ``rolling_corr`` returns NaN whenever either input is constant inside
    the window (denominator → 0). Several alpha factors then run the result
    through ``cs_rank`` (or another rank), where NaN poisons the per-day
    rank denominator. This wrapper substitutes NaN with null so that the
    downstream rank treats the row as missing instead. Mirrors STHSF's
    ``df.replace([inf,-inf], 0).fillna(0)`` pattern semantically (we map to
    null, which CS rank then ignores).
    """
    return ts_corr(x, y, window).fill_nan(None)

def ts_cov(x: pl.Expr, y: pl.Expr, window: int) -> pl.Expr:
    """Rolling sample covariance between two columns, per stock."""
    return pl.rolling_cov(x, y, window_size=window, min_samples=window).over(TS_PART)

def count_(col: pl.Expr, window: int) -> pl.Expr:
    """Rolling count of non-null values over ``window`` rows, per stock.

    Returned as :class:`pl.Float64`. ``min_samples=1`` so partial windows
    return the partial count (this matches the WorldQuant alphas that use
    ``count`` as a denominator).
    """
    indicator = col.is_not_null().cast(pl.Float64)
    return indicator.rolling_sum(window_size=window, min_samples=1).over(TS_PART)

def sumif(col: pl.Expr, cond: pl.Expr, window: int) -> pl.Expr:
    """Rolling sum of ``col`` where ``cond`` is True, over ``window`` rows.

    Equivalent to ``rolling_sum(where(cond, col, 0), window)``.
    """
    masked = pl.when(cond).then(col).otherwise(0.0)
    return masked.rolling_sum(window_size=window, min_samples=window).over(TS_PART)

def delay(col: pl.Expr, periods: int) -> pl.Expr:
    """Per-stock lag — equivalent to ``col.shift(periods)`` within each stock."""
    return col.shift(periods).over(TS_PART)

def delta(col: pl.Expr, periods: int) -> pl.Expr:
    """Per-stock difference — ``col - col.shift(periods)``."""
    return (col - col.shift(periods)).over(TS_PART)

def abs_(col: pl.Expr) -> pl.Expr:
    """Element-wise absolute value (suffix to avoid shadowing builtin ``abs``)."""
    return col.abs()

def log_(col: pl.Expr) -> pl.Expr:
    """Element-wise natural logarithm. Caller must ensure positivity."""
    return col.log()

def log1p(col: pl.Expr) -> pl.Expr:
    """Element-wise ``log(1 + x)``. Stable for small ``x``."""
    return col.log1p()

def sign_(col: pl.Expr) -> pl.Expr:
    """Element-wise sign: -1 / 0 / +1 (suffix to avoid shadowing)."""
    return col.sign()

def signed_power(col: pl.Expr, exponent: float) -> pl.Expr:
    """Sign-preserving power: ``sign(x) * |x|^exponent``."""
    return col.sign() * col.abs().pow(exponent)

def power(base: pl.Expr, exp: pl.Expr) -> pl.Expr:
    """Element-wise ``base ** exp`` for two expressions.

    For a constant exponent prefer :meth:`pl.Expr.pow` directly — this
    helper exists for symmetric two-Expr usage.
    """
    return base.pow(exp)

def clip_(col: pl.Expr, lo: float, hi: float) -> pl.Expr:
    """Element-wise clip into ``[lo, hi]``."""
    return col.clip(lower_bound=lo, upper_bound=hi)

def pmax(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """Element-wise (pairwise) maximum of two expressions.

    Wraps :func:`pl.max_horizontal`. NaN-tolerant — passes NaNs through.
    """
    return pl.max_horizontal(a, b)

def pmin(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """Element-wise (pairwise) minimum of two expressions.

    Wraps :func:`pl.min_horizontal`. NaN-tolerant — passes NaNs through.
    """
    return pl.min_horizontal(a, b)

def cs_rank(col: pl.Expr) -> pl.Expr:
    """Cross-section percentile rank in ``[0, 1]`` per ``trade_date``.

    Uses ``method='average'`` ranking divided by per-day non-null count,
    matching pandas ``rank(pct=True)`` and WorldQuant ``rank`` semantics.
    NaN is treated as missing (Polars otherwise ranks NaN as a value).
    """
    clean = col.fill_nan(None)
    return clean.rank(method='average').over(CS_PART) / clean.count().over(CS_PART)

def cs_scale(col: pl.Expr, scale: float=1.0) -> pl.Expr:
    """Per-day rescale so that ``sum(|x|) == scale``.

    WorldQuant ``scale(x, a)`` semantics: ``a * x / sum(|x|)`` per
    cross-section. Useful for portfolio-style alphas where you want unit
    gross exposure each day.
    """
    abs_sum = col.abs().sum().over(CS_PART)
    return col * scale / abs_sum

def ind_neutralize(col: pl.Expr, group: str | pl.Expr) -> pl.Expr:
    """Industry-neutralise: subtract per-day-per-group mean.

    ``group`` may be either a column name (``str``) or a pre-built
    :class:`pl.Expr`. The resulting series satisfies
    ``sum(out) ≈ 0`` within every ``(trade_date, group)`` cell.
    """
    group_expr = pl.col(group) if isinstance(group, str) else group
    return col - col.mean().over([pl.col(CS_PART), group_expr])

def if_then_else(cond: pl.Expr, then: pl.Expr | float, otherwise: pl.Expr | float) -> pl.Expr:
    """Convenience wrapper over ``pl.when(cond).then(then).otherwise(otherwise)``.

    Accepts either expressions or scalar literals for the two branches.
    """
    return pl.when(cond).then(then).otherwise(otherwise)
__all__ = ['TS_PART', 'CS_PART', 'ts_mean', 'ts_sum', 'ts_std', 'ts_min', 'ts_max', 'ts_median', 'ts_skew', 'ts_kurt', 'ts_zscore', 'ts_argmax', 'ts_argmin', 'ts_argmax_last', 'ts_argmin_last', 'ts_rank', 'ts_rank_int', 'ts_decay_linear', 'ts_product', 'ts_corr', 'ts_corr_safe', 'ts_cov', 'count_', 'sumif', 'sma', 'wma', 'delay', 'delta', 'abs_', 'log_', 'log1p', 'sign_', 'signed_power', 'power', 'clip_', 'pmax', 'pmin', 'cs_rank', 'cs_scale', 'ind_neutralize', 'if_then_else']

# ---- third_party/aurumq_gtja191/aurumq_rl/factors/gtja191/_ops.py ----
"""Shared polars operators for the GTJA-191 (国泰君安 191) factor library.

These operators are the GTJA-paper analogues of the alpha101 helpers in
``aurumq_rl.factors.alpha101._ops``. They share the same partition
conventions:

* Time-series operators partition by ``stock_code`` (assume the panel is
  sorted by ``[stock_code, trade_date]`` ascending).
* Cross-sectional operators partition by ``trade_date``.

All operators are pure :class:`polars.Expr` builders — they do not
eagerly evaluate. Compose with ``df.with_columns(...)``.

Semantics — important divergences from the WorldQuant/alpha101 conventions
--------------------------------------------------------------------------

* **GTJA SMA** (``sma(x, n, m)``) is **not** the simple moving average.
  In the GTJA paper, ``SMA(X, N, M)`` is recursively defined as
  ``today = (M * X_today + (N - M) * SMA_prev) / N``. This is exactly an
  exponentially weighted mean with smoothing factor ``alpha = M / N``,
  so we implement it via :meth:`pl.Expr.ewm_mean` with that alpha. (The
  Daic115 reference uses ``pandas.ewm(alpha=m/n)`` which is the same
  formulation.) The argument order is ``sma(col, n, m)`` and we require
  ``n > m`` (matching Daic115).

* **WMA** (``wma(x, n)``) uses **increasing** linear weights
  ``[1, 2, …, n] / sum``. The most recent observation gets weight ``n``,
  the oldest gets weight ``1``. Same direction as alpha101
  ``ts_decay_linear`` but with non-normalised weights (this normaliser
  matches Daic115's ``WMA``).

* **DECAYLINEAR** (``decay_linear(x, n)``) uses weights
  ``[2*i / (n*(n+1)) for i in 1..n]``. Sum is 1, direction matches
  ``wma`` (newest gets the highest weight). Algebraically identical to
  ``wma(x, n)`` because ``sum(1..n) == n*(n+1)/2`` — the two are
  preserved as separate names for fidelity to the GTJA paper.

* **HIGHDAY / LOWDAY** return the **distance in days** from today back
  to the high/low of the past N rows (so the latest row would yield 1 if
  the high/low is today). Range is ``[1, N]``. This matches Daic115
  ``HIGHDAY/LOWDAY`` (`n - argmax/argmin`).

* **REGBETA(y, x, n)** is the rolling slope of regressing y on x:
  ``cov(x, y) / var(x)`` over the last n rows. Argument order in our
  Python signature is ``(y, x, n)`` to match the natural "regress y on
  x" reading; Daic115's ``REGBETA(X, Y, N)`` is the same calculation
  but with positional names swapped — ours and theirs agree numerically.

* **REGRESI(y, x, n)** returns the residual at the **last point** of the
  rolling window: ``y[-1] - (alpha + beta * x[-1])`` where
  ``(alpha, beta)`` are the OLS estimates over the last n rows. Computed
  via :meth:`pl.Expr.rolling_map` + numpy ``lstsq`` — slow path.

Performance
-----------

Most operators are built from native polars rolling primitives and stay
vectorised: ``sma`` (via ``ewm_mean``), ``wma`` / ``decay_linear`` (via
weighted-shift composition), the ``ts_*`` rollers, ``lowday`` /
``highday`` (via shifted-window argmin/argmax), and ``regbeta`` (via
rolling cov / var). ``regresi`` remains the only slow-path rolling map.
"""
import polars as pl
TS_PART = 'stock_code'
'Partition column for time-series operators (rolling within stock).'
CS_PART = 'trade_date'
'Partition column for cross-sectional operators (rank within day).'

def rank(col: pl.Expr) -> pl.Expr:
    """GTJA ``RANK`` — cross-section percentile rank in ``[0, 1]`` per day.

    Uses ``method='average'`` ranking divided by the per-day count of
    non-null values, matching pandas ``rank(axis=1, pct=True)``.
    NaN is treated as missing.
    """
    clean = col.fill_nan(None)
    return clean.rank(method='average').over(CS_PART) / clean.count().over(CS_PART)

def ts_rank(col: pl.Expr, window: int) -> pl.Expr:
    """Rolling rank (pct) of the last value over the past ``window`` rows.

    Matches pandas ``rolling(N).rank(pct=True)``: returns the rank of the
    last observation in ``[1/n, 1]``. Null until the window is full or if
    any value in the window is null.
    """
    if window < 2:
        return pl.lit(0.0, dtype=pl.Float64)
    raw = col.rolling_rank(window_size=window, method='average', min_samples=window)
    return (raw / float(window)).over(TS_PART).cast(pl.Float64)

def mean(col: pl.Expr, window: int) -> pl.Expr:
    """Rolling arithmetic mean over ``window`` rows, per stock."""
    return col.rolling_mean(window_size=window, min_samples=window).over(TS_PART)

def sum_(col: pl.Expr, window: int) -> pl.Expr:
    """Rolling sum over ``window`` rows, per stock.

    Trailing underscore avoids shadowing Python's :func:`sum` builtin.
    """
    return col.rolling_sum(window_size=window, min_samples=window).over(TS_PART)

def std_(col: pl.Expr, window: int) -> pl.Expr:
    """Rolling sample stddev (ddof=1) over ``window`` rows, per stock.

    Trailing underscore avoids shadowing tools that expect ``std`` to be
    a class attribute (e.g. polars expression parsers).
    """
    return col.rolling_std(window_size=window, min_samples=window).over(TS_PART)

def ts_min(col: pl.Expr, window: int) -> pl.Expr:
    """Rolling minimum over ``window`` rows, per stock."""
    return col.rolling_min(window_size=window, min_samples=window).over(TS_PART)

def ts_max(col: pl.Expr, window: int) -> pl.Expr:
    """Rolling maximum over ``window`` rows, per stock."""
    return col.rolling_max(window_size=window, min_samples=window).over(TS_PART)

def delta(col: pl.Expr, periods: int) -> pl.Expr:
    """Per-stock difference: ``x - x.shift(periods)``."""
    return (col - col.shift(periods)).over(TS_PART)

def delay(col: pl.Expr, periods: int) -> pl.Expr:
    """Per-stock lag: ``x.shift(periods)``."""
    return col.shift(periods).over(TS_PART)

def corr(x: pl.Expr, y: pl.Expr, window: int) -> pl.Expr:
    """Rolling Pearson correlation between two columns, per stock."""
    return pl.rolling_corr(x, y, window_size=window, min_samples=window).over(TS_PART)

def covariance(x: pl.Expr, y: pl.Expr, window: int) -> pl.Expr:
    """Rolling sample covariance between two columns, per stock."""
    return pl.rolling_cov(x, y, window_size=window, min_samples=window).over(TS_PART)

def sma(col: pl.Expr, n: int, m: int=1) -> pl.Expr:
    """GTJA paper ``SMA(X, N, M)`` — recursive EWMA with smoothing ``M/N``.

    Defined by the recursion ``today = (M * X_today + (N - M) * SMA_prev) / N``,
    which is exactly :meth:`polars.Expr.ewm_mean` with ``alpha = M / N``.
    Daic115 implements this as ``pandas.DataFrame.ewm(alpha=M/N).mean()``,
    so our output matches theirs to within float-rounding.

    Parameters
    ----------
    col:
        Input series.
    n, m:
        Smoothing parameters; we require ``n > m > 0`` (Daic115's assertion).
    """
    if not n > m > 0:
        raise ValueError(f'sma requires n > m > 0, got n={n}, m={m}')
    alpha = float(m) / float(n)
    return col.ewm_mean(alpha=alpha, ignore_nulls=True).over(TS_PART).cast(pl.Float64)

def wma(col: pl.Expr, window: int) -> pl.Expr:
    """GTJA ``WMA`` — linear weighted MA with weights ``[1, 2, …, n] / sum``.

    Most recent observation gets the highest weight. Implemented as a
    sum of weighted ``shift(i)`` exprs (vectorised, no Python callback).
    Output is null for the first ``window - 1`` rows of each stock.
    """
    if window < 1:
        raise ValueError(f'window={window} must be >= 1')
    weights = [float(window - i) for i in range(window)]
    total = float(sum(weights))
    shifted = [col.shift(i).over(TS_PART) * (w / total) for i, w in enumerate(weights)]
    out = shifted[0]
    for term in shifted[1:]:
        out = out + term
    return out.cast(pl.Float64)

def decay_linear(col: pl.Expr, window: int) -> pl.Expr:
    """GTJA ``DECAYLINEAR`` — weights ``[2*i / (n*(n+1)) for i in 1..n]``.

    Mathematically identical to :func:`wma` (since the sum of 1..n is
    n*(n+1)/2, both produce the same normalised increasing-weight MA).
    Kept as a separate symbol for fidelity to the GTJA paper.
    """
    return wma(col, window)

def ifelse(cond: pl.Expr, then: pl.Expr | float, otherwise: pl.Expr | float) -> pl.Expr:
    """GTJA ``IFELSE(cond, A, B)`` — ``pl.when(cond).then(A).otherwise(B)``."""
    return pl.when(cond).then(then).otherwise(otherwise)

def count_(cond: pl.Expr, window: int) -> pl.Expr:
    """Rolling count of ``True`` values of a boolean condition.

    Trailing underscore avoids shadowing Python's :func:`count` (none in
    builtins, but consistent with ``sum_`` / ``std_``).
    """
    indicator = cond.cast(pl.Float64)
    return indicator.rolling_sum(window_size=window, min_samples=window).over(TS_PART)

def sumif(col: pl.Expr, window: int, cond: pl.Expr) -> pl.Expr:
    """Rolling sum of ``col`` where ``cond`` is ``True``, over ``window`` rows.

    Equivalent to ``rolling_sum(where(cond, col, 0), window)``.
    """
    masked = pl.when(cond).then(col).otherwise(0.0)
    return masked.rolling_sum(window_size=window, min_samples=window).over(TS_PART)

def _window_values(col: pl.Expr, window: int) -> list[pl.Expr]:
    """Return oldest->newest values in the current rolling window."""
    return [col.shift(window - 1 - i).over(TS_PART) for i in range(window)]

def _window_valid(values: list[pl.Expr]) -> pl.Expr:
    """True when a shifted rolling window contains no null or NaN values."""
    return pl.all_horizontal([value.is_not_null() & ~value.is_nan() for value in values])

def _window_arg_extreme(col: pl.Expr, window: int, *, kind: str) -> pl.Expr:
    """Vectorized rolling argmax/argmin with first-tie semantics."""
    if window < 1:
        raise ValueError(f'window={window} must be >= 1')
    values = _window_values(col, window)
    valid = _window_valid(values)
    extreme = pl.max_horizontal(values) if kind == 'max' else pl.min_horizontal(values)
    out = pl.lit(None, dtype=pl.Float64)
    for idx in range(window - 1, -1, -1):
        out = pl.when(values[idx] == extreme).then(float(idx)).otherwise(out)
    return pl.when(valid).then(out).otherwise(None).cast(pl.Float64)

def lowday(col: pl.Expr, window: int) -> pl.Expr:
    """GTJA ``LOWDAY`` — distance from today to the min of the past N rows.

    Returns ``n - argmin(window)`` so that "today is the new low" yields
    ``1`` and "the low was n-1 rows ago" yields ``n``. Matches Daic115's
    ``LOWDAY``. Output is :class:`pl.Float64` (NaN for partial windows).
    """
    return (float(window) - _window_arg_extreme(col, window, kind='min')).cast(pl.Float64)

def highday(col: pl.Expr, window: int) -> pl.Expr:
    """GTJA ``HIGHDAY`` — distance from today to the max of the past N rows.

    Range ``[1, n]``. See :func:`lowday` for the convention.
    """
    return (float(window) - _window_arg_extreme(col, window, kind='max')).cast(pl.Float64)

def regbeta(y: pl.Expr, x: pl.Expr, window: int) -> pl.Expr:
    """Rolling OLS slope of ``y`` on ``x`` over the last ``window`` rows.

    ``slope = cov(x, y) / var(x)``. Built from native rolling cov/var so
    it stays vectorised. Returns null until the window is full.

    Notes
    -----
    Argument order ``(y, x, n)`` reads as "regress y on x". Daic115's
    ``REGBETA(X, Y, N)`` computes ``cov(X, Y) / var(X)`` — the same number
    with the X/Y names swapped at the call site. Numerical agreement
    holds.
    """
    cov_xy = pl.rolling_cov(x, y, window_size=window, min_samples=window).over(TS_PART)
    var_x = x.rolling_var(window_size=window, min_samples=window).over(TS_PART)
    safe_var = pl.when(var_x == 0).then(None).otherwise(var_x)
    return (cov_xy / safe_var).cast(pl.Float64)

def regresi(y: pl.Expr, x: pl.Expr, window: int) -> pl.Expr:
    """Rolling OLS residual of ``y`` on ``x`` at the **last** window point.

    For each window of ``window`` rows ending at the current row, fit
    ``y = a + b*x`` and return ``y[-1] - (a + b*x[-1])``.

    Implemented via the closed-form OLS identity (vectorised, no Python
    callback). Algebra:

    .. code-block:: text

        b      = cov(x, y) / var(x)
        a      = mean(y) - b * mean(x)
        resid  = y[-1] - (a + b * x[-1])
               = (y[-1] - mean(y)) - b * (x[-1] - mean(x))

    so the residual is ``y_dev_today - beta * x_dev_today`` where the
    deviations are from the rolling mean.
    """
    cov_xy = pl.rolling_cov(x, y, window_size=window, min_samples=window).over(TS_PART)
    var_x = x.rolling_var(window_size=window, min_samples=window).over(TS_PART)
    safe_var = pl.when(var_x == 0).then(None).otherwise(var_x)
    beta = cov_xy / safe_var
    mean_x = x.rolling_mean(window_size=window, min_samples=window).over(TS_PART)
    mean_y = y.rolling_mean(window_size=window, min_samples=window).over(TS_PART)
    return (y - mean_y - beta * (x - mean_x)).cast(pl.Float64)

def sign_(col: pl.Expr) -> pl.Expr:
    """Element-wise sign: ``-1 / 0 / +1``. Trailing underscore avoids shadowing."""
    return col.sign()

def abs_(col: pl.Expr) -> pl.Expr:
    """Element-wise absolute value."""
    return col.abs()

def log_(col: pl.Expr) -> pl.Expr:
    """Element-wise natural log. Caller must ensure positivity."""
    return col.log()

def sequence(n: int) -> pl.Expr:
    """Return the literal expression representing the constant series ``[1, 2, …, n]``.

    Used as a deterministic ``x`` argument to :func:`regbeta` / :func:`regresi`
    when the GTJA formula calls ``REGBETA(SEQUENCE(N), …)``. The returned
    expression is a polars ``Series literal`` so it broadcasts correctly
    when joined with another expression of the same length.
    """
    if n < 1:
        raise ValueError(f'sequence requires n >= 1, got {n}')
    return pl.lit(pl.Series('__seq__', list(range(1, n + 1)), dtype=pl.Float64))
__all__ = ['TS_PART', 'CS_PART', 'rank', 'ts_rank', 'mean', 'sum_', 'std_', 'ts_min', 'ts_max', 'delta', 'delay', 'corr', 'covariance', 'sma', 'wma', 'decay_linear', 'ifelse', 'count_', 'sumif', 'lowday', 'highday', 'regbeta', 'regresi', 'sign_', 'abs_', 'log_', 'sequence']

# ---- third_party/aurumq_gtja191/aurumq_rl/factors/gtja191/batch_001_020.py ----
"""GTJA-191 factor library — batch 001 through 020.

Each function takes a sorted (``stock_code``, ``trade_date``) panel
:class:`pl.DataFrame` and returns a :class:`pl.Series` aligned to its rows.

Formulas are translated from the Daic115/alpha191 reference (no LICENSE,
formula-only reference — code is NOT vendored). The Guotai Junan 2017
paper (refs/gtja191/wpwp__Alpha-101-GTJA-191/) is the original source.

When a polars expression mixes a TS partition (``stock_code``) and a CS
partition (``trade_date``), we materialise the inner result with
``with_columns(...)`` before applying the outer partition — polars
cannot reliably nest different ``over(...)`` partitions inside one
expression.
"""
import polars as pl

def gtja_001(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #001 — Volume change rank vs intraday return correlation.

    Guotai Junan Formula
    --------------------
        (-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))

    Reference: Daic115/alpha191 alpha191_001 (formula only)

    Required panel columns: ``volume``, ``close``, ``open``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    rv = delta(log_(pl.col('volume')), 1)
    rret = (pl.col('close') - pl.col('open')) / pl.col('open')
    staged = panel.with_columns(rv.alias('__g001_dlv'), rret.alias('__g001_ret'))
    staged = staged.with_columns(rank(pl.col('__g001_dlv')).alias('__g001_rv'), rank(pl.col('__g001_ret')).alias('__g001_rret'))
    return staged.select((-1.0 * corr(pl.col('__g001_rv'), pl.col('__g001_rret'), 6)).alias('gtja_001').cast(pl.Float64)).to_series()

def gtja_002(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #002 — One-period delta of normalised mid-range.

    Guotai Junan Formula
    --------------------
        (-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))

    Required panel columns: ``close``, ``low``, ``high``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    base = (pl.col('close') - pl.col('low') - (pl.col('high') - pl.col('close'))) / (pl.col('high') - pl.col('low'))
    return panel.select((-1.0 * delta(base, 1)).alias('gtja_002').cast(pl.Float64)).to_series()

def gtja_003(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #003 — 6-day sum of close-vs-extreme conditional flow.

    Guotai Junan Formula
    --------------------
        SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?
             MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)

    Required panel columns: ``close``, ``low``, ``high``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    delay1 = delay(pl.col('close'), 1)
    cond_up = pl.col('close') > delay1
    cond_dn = pl.col('close') < delay1
    pivot = ifelse(cond_up, pl.min_horizontal(pl.col('low'), delay1), pl.max_horizontal(pl.col('high'), delay1))
    inner = pl.when(cond_up | cond_dn).then(pl.col('close') - pivot).otherwise(0.0)
    return panel.select(sum_(inner, 6).alias('gtja_003').cast(pl.Float64)).to_series()

def gtja_004(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #004 — Trend regime conditional with volume gate.

    Guotai Junan Formula
    --------------------
        if((MEAN(CLOSE,8)+STD(CLOSE,8))<MEAN(CLOSE,2)) -1
        elif(MEAN(CLOSE,2)<(MEAN(CLOSE,8)-STD(CLOSE,8))) 1
        elif(VOLUME/MEAN(VOLUME,20) >= 1) 1
        else -1

    Required panel columns: ``close``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``mean_reversion``
    """
    m8 = mean(pl.col('close'), 8)
    s8 = std_(pl.col('close'), 8)
    m2 = mean(pl.col('close'), 2)
    vol_ratio = pl.col('volume') / mean(pl.col('volume'), 20)
    expr = pl.when(m8 + s8 < m2).then(-1.0).otherwise(pl.when(m2 < m8 - s8).then(1.0).otherwise(pl.when(vol_ratio >= 1.0).then(1.0).otherwise(-1.0)))
    return panel.select(expr.alias('gtja_004').cast(pl.Float64)).to_series()

def gtja_005(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #005 — Negated 3d rolling-max of 5d ts-rank corr.

    Guotai Junan Formula
    --------------------
        (-1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3))

    Reference parquet does not include gtja_005 because Daic115 used
    pandas ``rolling.rank`` whose semantics changed across versions.
    Our implementation uses our own ``ts_rank`` (rank of last value in
    the window, polars rolling_rank); reference test will be skipped.

    Required panel columns: ``volume``, ``high``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged = panel.with_columns(ts_rank(pl.col('volume'), 5).alias('__g005_tv'), ts_rank(pl.col('high'), 5).alias('__g005_th'))
    staged = staged.with_columns(corr(pl.col('__g005_tv'), pl.col('__g005_th'), 5).alias('__g005_c'))
    return staged.select((-1.0 * ts_max(pl.col('__g005_c'), 3)).alias('gtja_005').cast(pl.Float64)).to_series()

def gtja_006(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #006 — Negated rank of sign of 4d weighted O/H delta.

    Guotai Junan Formula
    --------------------
        (RANK(SIGN(DELTA((OPEN * 0.85 + HIGH * 0.15), 4))) * -1)

    Required panel columns: ``open``, ``high``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    val = pl.col('open') * 0.85 + pl.col('high') * 0.15
    sgn = sign_(delta(val, 4))
    staged = panel.with_columns(sgn.alias('__g006_sgn'))
    return staged.select((-1.0 * rank(pl.col('__g006_sgn'))).alias('gtja_006').cast(pl.Float64)).to_series()

def gtja_007(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #007 — VWAP-close 3d max+min ranks × volume-delta rank.

    Guotai Junan Formula
    --------------------
        (RANK(MAX(VWAP - CLOSE, 3)) + RANK(MIN(VWAP - CLOSE, 3))) *
        RANK(DELTA(VOLUME, 3))

    Required panel columns: ``vwap``, ``close``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    diff = pl.col('vwap') - pl.col('close')
    staged = panel.with_columns(ts_max(diff, 3).alias('__g007_max'), ts_min(diff, 3).alias('__g007_min'), delta(pl.col('volume'), 3).alias('__g007_dv'))
    return staged.select(((rank(pl.col('__g007_max')) + rank(pl.col('__g007_min'))) * rank(pl.col('__g007_dv'))).alias('gtja_007').cast(pl.Float64)).to_series()

def gtja_008(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #008 — Negated rank of 4d delta of HL10+VWAP80 weighted price.

    Guotai Junan Formula
    --------------------
        RANK(DELTA(((HIGH+LOW)/2)*0.2 + VWAP*0.8, 4) * -1)

    Daic115 wrote this as ``-1*(H+L)*0.1 + VWAP*0.8`` (different op
    precedence) which differs from the spec. We follow Daic115 for
    parity with the reference parquet.

    Required panel columns: ``high``, ``low``, ``vwap``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    val = -1.0 * (pl.col('high') + pl.col('low')) * 0.1 + pl.col('vwap') * 0.8
    staged = panel.with_columns(delta(val, 4).alias('__g008_d'))
    return staged.select(rank(pl.col('__g008_d')).alias('gtja_008').cast(pl.Float64)).to_series()

def gtja_009(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #009 — EWMA(7,2) of mid-price acceleration weighted by HL/Volume.

    Guotai Junan Formula
    --------------------
        SMA(((H+L)/2 - (DELAY(H,1)+DELAY(L,1))/2) * (H-L)/VOLUME, 7, 2)

    Required panel columns: ``high``, ``low``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    mid = (pl.col('high') + pl.col('low')) / 2.0
    mid_prev = (delay(pl.col('high'), 1) + delay(pl.col('low'), 1)) / 2.0
    range_per_vol = (pl.col('high') - pl.col('low')) / pl.col('volume')
    inner = (mid - mid_prev) * range_per_vol
    return panel.select(sma(inner, 7, 2).alias('gtja_009').cast(pl.Float64)).to_series()

def gtja_010(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #010 — Rank of conditional 20d-return-std-or-close squared.

    Guotai Junan Formula
    --------------------
        (RANK(MAX(((RET<0)?STD(RET,20):CLOSE)^2),5))

    Daic115 implementation uses ``np.maximum(alpha, 5)`` which is an
    elementwise scalar floor (probably an upstream bug — should be
    ``ts_max(alpha, 5)``). We match Daic115 for reference parity.

    Required panel columns: ``close``, ``returns``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volatility``
    """
    ret = pl.col('returns')
    cond_branch = ifelse(ret < 0.0, std_(ret, 20), pl.col('close'))
    inner = cond_branch * cond_branch
    floored = pl.max_horizontal(
        inner.cast(pl.Float64), pl.lit(5.0, dtype=pl.Float64)
    )
    staged = panel.with_columns(floored.alias('__g010_inner'))
    return staged.select(rank(pl.col('__g010_inner')).alias('gtja_010').cast(pl.Float64)).to_series()

def gtja_011(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #011 — 6-day sum of normalised mid-range times volume.

    Guotai Junan Formula
    --------------------
        SUM((2*CLOSE - LOW - HIGH) / (HIGH - LOW) * VOLUME, 6)

    Required panel columns: ``close``, ``low``, ``high``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    base = (2.0 * pl.col('close') - pl.col('low') - pl.col('high')) / (pl.col('high') - pl.col('low'))
    return panel.select(sum_(base * pl.col('volume'), 6).alias('gtja_011').cast(pl.Float64)).to_series()

def gtja_012(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #012 — Rank(O - MA(VWAP,10)) * -1 * Rank(|C - VWAP|).

    Guotai Junan Formula
    --------------------
        RANK(OPEN - SUM(VWAP,10)/10) * (-1 * RANK(ABS(CLOSE - VWAP)))

    Required panel columns: ``open``, ``vwap``, ``close``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    staged = panel.with_columns((pl.col('open') - mean(pl.col('vwap'), 10)).alias('__g012_o'), abs_(pl.col('close') - pl.col('vwap')).alias('__g012_d'))
    return staged.select((rank(pl.col('__g012_o')) * (-1.0 * rank(pl.col('__g012_d')))).alias('gtja_012').cast(pl.Float64)).to_series()

def gtja_013(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #013 — Geometric mean of (H, L) minus VWAP.

    Guotai Junan Formula
    --------------------
        (HIGH * LOW)^0.5 - VWAP

    Required panel columns: ``high``, ``low``, ``vwap``.

    Direction: ``normal``
    Category: ``mean_reversion``
    """
    expr = (pl.col('high') * pl.col('low')) ** 0.5 - pl.col('vwap')
    return panel.select(expr.alias('gtja_013').cast(pl.Float64)).to_series()

def gtja_014(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #014 — 5-day price change.

    Guotai Junan Formula
    --------------------
        CLOSE - DELAY(CLOSE, 5)

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    return panel.select(delta(pl.col('close'), 5).alias('gtja_014').cast(pl.Float64)).to_series()

def gtja_015(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #015 — Overnight gap return (open / prior close - 1).

    Guotai Junan Formula
    --------------------
        OPEN / DELAY(CLOSE, 1) - 1

    Required panel columns: ``open``, ``close``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    expr = pl.col('open') / delay(pl.col('close'), 1) - 1.0
    return panel.select(expr.alias('gtja_015').cast(pl.Float64)).to_series()

def gtja_016(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #016 — Negated 5d max of rank(volume,vwap)-corr rank.

    Guotai Junan Formula
    --------------------
        -1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5)

    Required panel columns: ``volume``, ``vwap``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged = panel.with_columns(rank(pl.col('volume')).alias('__g016_rv'), rank(pl.col('vwap')).alias('__g016_rw'))
    staged = staged.with_columns(corr(pl.col('__g016_rv'), pl.col('__g016_rw'), 5).alias('__g016_c'))
    staged = staged.with_columns(rank(pl.col('__g016_c')).alias('__g016_r'))
    return staged.select((-1.0 * ts_max(pl.col('__g016_r'), 5)).alias('gtja_016').cast(pl.Float64)).to_series()

def gtja_017(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #017 — Rank(VWAP - 15d-max-VWAP) raised to 5d close delta.

    Guotai Junan Formula
    --------------------
        RANK(VWAP - MAX(VWAP, 15)) ^ DELTA(CLOSE, 5)

    Numerical safety
    ----------------
    The raw formula has ``rank ∈ [0, 1] ^ delta(close, 5)``. When the
    exponent gets large in absolute value (delta(close, 5) reaches ±50
    on adj_factor-glitchy days), the result blows up to 1e+308 and float
    cast to inf. We use ``safe_pow_clip`` which clips the exponent to
    [-3, 3] — bounded growth/decay rate, no overflow. Economic intent
    preserved (a 3-fold price move is the absolute ceiling that matters
    for momentum). Previous behaviour produced 4621 inf cells/year on
    real data (max_finite = 1.4e+308); this fix eliminates them.

    Required panel columns: ``vwap``, ``close``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    inner = pl.col('vwap') - ts_max(pl.col('vwap'), 15)
    staged = panel.with_columns(inner.alias('__g017_inner'))
    staged = staged.with_columns(rank(pl.col('__g017_inner')).alias('__g017_r'))
    return staged.select(safe_pow_clip(pl.col('__g017_r'), delta(pl.col('close'), 5)).alias('gtja_017').cast(pl.Float64)).to_series()

def gtja_018(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #018 — 5d price ratio (close / delay(close, 5)).

    Guotai Junan Formula
    --------------------
        CLOSE / DELAY(CLOSE, 5)

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    expr = pl.col('close') / delay(pl.col('close'), 5)
    return panel.select(expr.alias('gtja_018').cast(pl.Float64)).to_series()

def gtja_019(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #019 — Asymmetric 6-day price change ratio (vwap-anchored).

    Guotai Junan Formula
    --------------------
        if (VWAP < DELAY(VWAP, 6)) (VWAP - DELAY(VWAP, 6)) / DELAY(VWAP, 6)
        elif VWAP == DELAY(VWAP, 6) 0
        else (VWAP - DELAY(VWAP, 6)) / VWAP

    Daic115 uses VWAP by default (use_vwap=True). We follow that.

    Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``mean_reversion``
    """
    vwap = pl.col('vwap')
    vwap_lag = delay(vwap, 6)
    diff = vwap - vwap_lag
    branch_down = diff / vwap_lag
    branch_up = diff / vwap
    expr = pl.when(vwap < vwap_lag).then(branch_down).otherwise(branch_up)
    return panel.select(expr.alias('gtja_019').cast(pl.Float64)).to_series()

def gtja_020(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #020 — 6-day % change times 100 (vwap-anchored).

    Guotai Junan Formula
    --------------------
        (VWAP - DELAY(VWAP, 6)) / DELAY(VWAP, 6) * 100

    Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    vwap = pl.col('vwap')
    vwap_lag = delay(vwap, 6)
    expr = (vwap - vwap_lag) / vwap_lag * 100.0
    return panel.select(expr.alias('gtja_020').cast(pl.Float64)).to_series()
_DOC_BASE = 'docs/factor_library/gtja191'
_REF_BASE = "Guotai Junan 2017, '191 Alphas', via Daic115/alpha191 (formula only)"
_ENTRIES: list[FactorEntry] = [FactorEntry(id='gtja_001', impl=gtja_001, direction='reverse', category='volume_price', description='Volume-change rank vs intraday return rank correlation, 6d, negated', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_001.md'), FactorEntry(id='gtja_002', impl=gtja_002, direction='reverse', category='mean_reversion', description='One-day delta of normalised intraday mid-range, negated', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_002.md'), FactorEntry(id='gtja_003', impl=gtja_003, direction='normal', category='volume_price', description='6d sum of close-vs-extreme conditional flow', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_003.md'), FactorEntry(id='gtja_004', impl=gtja_004, direction='normal', category='mean_reversion', description='Trend regime + volume gate ternary signal', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_004.md'), FactorEntry(id='gtja_005', impl=gtja_005, direction='reverse', category='volume_price', description='Negated 3d max of 5d ts-rank corr (volume vs high)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_005.md', quality_flag=1), FactorEntry(id='gtja_006', impl=gtja_006, direction='reverse', category='momentum', description='Rank of sign of 4d weighted (open*0.85 + high*0.15) delta, negated', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_006.md'), FactorEntry(id='gtja_007', impl=gtja_007, direction='normal', category='volume_price', description='(Rank max + Rank min) of (vwap-close,3) × Rank(volume delta,3)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_007.md'), FactorEntry(id='gtja_008', impl=gtja_008, direction='reverse', category='momentum', description='Negated rank of 4d delta of HL10+VWAP80 weighted price (Daic115 parity)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_008.md'), FactorEntry(id='gtja_009', impl=gtja_009, direction='normal', category='volume_price', description='EWMA(7,2) of mid-price acceleration weighted by (H-L)/Volume', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_009.md'), FactorEntry(id='gtja_010', impl=gtja_010, direction='reverse', category='volatility', description='Rank of MAX5(((ret<0?std20:close))^2) — Daic115 floor-by-5 parity', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_010.md'), FactorEntry(id='gtja_011', impl=gtja_011, direction='normal', category='volume_price', description='6d sum of normalised mid-range × volume', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_011.md'), FactorEntry(id='gtja_012', impl=gtja_012, direction='reverse', category='mean_reversion', description='Rank(O-MA10VWAP) × -Rank(|C-VWAP|)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_012.md'), FactorEntry(id='gtja_013', impl=gtja_013, direction='normal', category='mean_reversion', description='Geometric mean of (H, L) minus VWAP', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_013.md'), FactorEntry(id='gtja_014', impl=gtja_014, direction='normal', category='momentum', description='5d price change (close - delay(close,5))', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_014.md'), FactorEntry(id='gtja_015', impl=gtja_015, direction='normal', category='momentum', description='Overnight gap return: open / prior close - 1', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_015.md'), FactorEntry(id='gtja_016', impl=gtja_016, direction='reverse', category='volume_price', description='-1 × TSMAX(rank(corr(rank(vol), rank(vwap), 5)), 5)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_016.md'), FactorEntry(id='gtja_017', impl=gtja_017, direction='reverse', category='momentum', description='Rank(vwap - max15(vwap)) ^ delta(close, 5)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_017.md'), FactorEntry(id='gtja_018', impl=gtja_018, direction='normal', category='momentum', description='5d close ratio: close / delay(close, 5)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_018.md'), FactorEntry(id='gtja_019', impl=gtja_019, direction='normal', category='mean_reversion', description='Asymmetric 6d vwap change ratio', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_019.md'), FactorEntry(id='gtja_020', impl=gtja_020, direction='normal', category='momentum', description='6d vwap % change × 100', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_020.md')]
for _e in _ENTRIES:
    register_gtja191(_e)

# ---- third_party/aurumq_gtja191/aurumq_rl/factors/gtja191/batch_021_040.py ----
"""GTJA-191 factor library — batch 021 through 040.

Translated from Daic115/alpha191 (formula reference, no code vendored).
"""
import polars as pl

def _stock_row_index(panel: pl.DataFrame) -> pl.Expr:
    """Per-stock cumulative row counter (0-based), expressed as Float64.

    ``regbeta(y, _stock_row_index(...), n)`` returns the rolling slope of
    ``y`` against ``[k, k+1, …, k+n-1]``, which (by translation
    invariance of covariance) equals the slope against the natural
    SEQUENCE(1..n) used in the Guotai Junan paper.
    """
    return pl.int_range(pl.len(), dtype=pl.Int64).over('stock_code').cast(pl.Float64)

def gtja_021(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #021 — Rolling 6d slope of MEAN(close, 6) vs sequence.

    Guotai Junan Formula
    --------------------
        REGBETA(MEAN(CLOSE, 6), SEQUENCE(6))

    Daic115 uses ``qlib.data.ops.rolling_slope`` (cython). We compute
    the same number via ``regbeta(y, x, 6)`` where ``x`` is a per-stock
    row counter — the rolling slope is invariant to additive shifts of
    ``x``. Reference parquet does NOT include gtja_021 because the
    Daic115 builder skipped qlib-dependent alphas; reference test is
    skipped.

    Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    val_mean = mean(pl.col('vwap'), 6)
    staged = panel.with_columns(val_mean.alias('__g021_y'), _stock_row_index(panel).alias('__g021_x'))
    return staged.select(regbeta(pl.col('__g021_y'), pl.col('__g021_x'), 6).alias('gtja_021').cast(pl.Float64)).to_series()

def gtja_022(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #022 — EWMA(12,1) of (close mean-detrend - lag3).

    Guotai Junan Formula
    --------------------
        SMA((C - MEAN(C,6))/MEAN(C,6) -
             DELAY((C - MEAN(C,6))/MEAN(C,6), 3), 12, 1)

    Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    vwap = pl.col('vwap')
    val_mean = mean(vwap, 6)
    detrend = (vwap - val_mean) / val_mean
    diff = detrend - delay(detrend, 3)
    return panel.select(sma(diff, 12, 1).alias('gtja_022').cast(pl.Float64)).to_series()

def gtja_023(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #023 — Up-day std share over 20d, smoothed by SMA(20,1).

    Guotai Junan Formula
    --------------------
        SMA(cond ? STD(C,20) : 0, 20, 1) /
        (SMA(cond ? STD(C,20) : 0, 20, 1) + SMA(!cond ? STD(C,20) : 0, 20, 1)) * 100
        where cond = C > DELAY(C, 1)

    Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volatility``
    """
    vwap = pl.col('vwap')
    cond = vwap > delay(vwap, 1)
    s = std_(vwap, 20)
    up = ifelse(cond, s, 0.0)
    dn = ifelse(~cond, s, 0.0)
    sma_up = sma(up, 20, 1)
    sma_dn = sma(dn, 20, 1)
    expr = sma_up / (sma_up + sma_dn) * 100.0
    return panel.select(expr.alias('gtja_023').cast(pl.Float64)).to_series()

def gtja_024(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #024 — EWMA(5,1) of 5d price change.

    Guotai Junan Formula
    --------------------
        SMA(CLOSE - DELAY(CLOSE, 5), 5, 1)

    Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    vwap = pl.col('vwap')
    diff = vwap - delay(vwap, 5)
    return panel.select(sma(diff, 5, 1).alias('gtja_024').cast(pl.Float64)).to_series()

def gtja_025(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #025 — Composite of close-delta rank, volume EWMA rank, return-sum rank.

    Guotai Junan Formula
    --------------------
        (-1 * RANK(DELTA(CLOSE, 7) * (1 - RANK(DECAYLINEAR(VOLUME / MEAN(VOLUME, 20), 9))))) *
        (1 + RANK(SUM(RET, 250)))

    Daic115 uses ``ewm(alpha=1/9)`` for the decay step (not the linear-
    weighted DECAYLINEAR), and `period=150` instead of 250 by default.
    We follow Daic115 (period=150, ewma instead of decay_linear).

    Required panel columns: ``vwap``, ``volume``, ``returns``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    vwap = pl.col('vwap')
    ret_sum = sum_(pl.col('returns'), 150)
    vol_ratio = pl.col('volume') / mean(pl.col('volume'), 20)
    vol_ewma = sma(vol_ratio, 9, 1)
    staged = panel.with_columns(delta(vwap, 7).alias('__g025_d7'), vol_ewma.alias('__g025_ve'), ret_sum.alias('__g025_rs'))
    staged = staged.with_columns(rank(pl.col('__g025_d7')).alias('__g025_rd'), rank(pl.col('__g025_ve')).alias('__g025_rv'), rank(pl.col('__g025_rs')).alias('__g025_rr'))
    expr = -1.0 * pl.col('__g025_rd') * pl.col('__g025_rv') * (1.0 + pl.col('__g025_rr'))
    return staged.select(expr.alias('gtja_025').cast(pl.Float64)).to_series()

def gtja_026(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #026 — Long-window VWAP/close-lag correlation + mean reversion.

    Guotai Junan Formula
    --------------------
        (MEAN(CLOSE, 7) - CLOSE) + CORR(VWAP, DELAY(CLOSE, 5), 230)

    Daic115 default: ma_period=12, corr_period=200.

    Required panel columns: ``close``, ``vwap``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``mean_reversion``
    """
    expr = mean(pl.col('close'), 12) - pl.col('close') + corr(pl.col('vwap'), delay(pl.col('close'), 5), 200)
    return panel.select(expr.alias('gtja_026').cast(pl.Float64)).to_series()

def gtja_027(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #027 — EWMA(12,1) of 3d+6d % momentum sum.

    Guotai Junan Formula
    --------------------
        WMA(((C/DELAY(C,3) - 1)*100 + (C/DELAY(C,6) - 1)*100), 12)

    Daic115 substitutes WMA with SMA(.,12,1) — we follow that.

    Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    vwap = pl.col('vwap')
    short = (vwap / delay(vwap, 3) - 1.0) * 100.0
    long_ = (vwap / delay(vwap, 6) - 1.0) * 100.0
    return panel.select(sma(short + long_, 12, 1).alias('gtja_027').cast(pl.Float64)).to_series()

def gtja_028(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #028 — KDJ-style 9d stochastic with two SMA layers.

    Guotai Junan Formula
    --------------------
        3 * SMA((C - TSMIN(L,9)) / (TSMAX(H,9) - TSMIN(L,9)) * 100, 3, 1) -
        2 * SMA(SMA((C - TSMIN(L,9)) / (TSMAX(H,9) - TSMIN(L,9)) * 100, 3, 1), 3, 1)

    Required panel columns: ``close``, ``low``, ``high``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    low_min = ts_min(pl.col('low'), 9)
    high_max = ts_max(pl.col('high'), 9)
    raw = (pl.col('close') - low_min) / (high_max - low_min) * 100.0
    sma1 = sma(raw, 3, 1)
    sma2 = sma(sma1, 3, 1)
    return panel.select((3.0 * sma1 - 2.0 * sma2).alias('gtja_028').cast(pl.Float64)).to_series()

def gtja_029(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #029 — 6d % change × log(volume).

    Guotai Junan Formula
    --------------------
        (CLOSE - DELAY(CLOSE, 6)) / DELAY(CLOSE, 6) * VOLUME

    Daic115 uses ``log(volume)`` rather than raw volume — we follow.

    Required panel columns: ``vwap``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    vwap = pl.col('vwap')
    expr = (vwap - delay(vwap, 6)) / delay(vwap, 6) * pl.col('volume').log()
    return panel.select(expr.alias('gtja_029').cast(pl.Float64)).to_series()

def gtja_030(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #030 — STUB (Fama-French residual^2 WMA).

    Guotai Junan Formula
    --------------------
        WMA((REGRESI(CLOSE/DELAY(CLOSE)-1, MKT, SMB, HML, 60))^2, 20)

    Daic115 marks this as ``unfinished=True`` and returns ``None``.
    A faithful implementation requires Fama-French factor exposures
    which the synthetic panel does not provide. We return an all-NaN
    series and tag with quality_flag=2 (stub). Reference parquet does
    not include gtja_030; reference test is skipped.

    Direction: ``normal``
    Category: ``volatility``
    """
    return panel.select((pl.col('close') * 0.0 + float('nan')).alias('gtja_030').cast(pl.Float64)).to_series()

def gtja_031(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #031 — 12d mean-distance ratio × 100 (vwap-anchored).

    Guotai Junan Formula
    --------------------
        (CLOSE - MEAN(CLOSE, 12)) / MEAN(CLOSE, 12) * 100

    Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    vwap = pl.col('vwap')
    m12 = mean(vwap, 12)
    return panel.select(((vwap - m12) / m12 * 100.0).alias('gtja_031').cast(pl.Float64)).to_series()

def gtja_032(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #032 — Negated 3d sum of rank(corr(rank-H, rank-Vol, 3)).

    Guotai Junan Formula
    --------------------
        -1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3)

    Required panel columns: ``high``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged = panel.with_columns(rank(pl.col('high')).alias('__g032_rh'), rank(pl.col('volume')).alias('__g032_rv'))
    staged = staged.with_columns(corr(pl.col('__g032_rh'), pl.col('__g032_rv'), 3).alias('__g032_c'))
    staged = staged.with_columns(rank(pl.col('__g032_c')).alias('__g032_rc'))
    return staged.select((-1.0 * sum_(pl.col('__g032_rc'), 3)).alias('gtja_032').cast(pl.Float64)).to_series()

def gtja_033(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #033 — Long/medium return spread × low-min change × turnover rank.

    Guotai Junan Formula
    --------------------
        (((-1 * TSMIN(LOW, 5)) + DELAY(TSMIN(LOW, 5), 5)) *
         RANK((SUM(RET, 240) - SUM(RET, 20)) / 220)) * TSRANK(VOLUME, 5)

    Daic115 references ``data["turn"]`` (turnover_rate) which we don't
    have. Use ``amount / cap`` as proxy. Reference parquet does NOT
    include gtja_033; reference test is skipped.

    Required panel columns: ``low``, ``returns``, ``amount``, ``cap``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    low_min = ts_min(pl.col('low'), 5)
    ret = pl.col('returns')
    ret_sum = sum_(ret, 240) - sum_(ret, 20)
    turn_proxy = pl.col('amount') / pl.col('cap')
    staged = panel.with_columns(ret_sum.alias('__g033_rs'), turn_proxy.alias('__g033_tp'))
    staged = staged.with_columns(rank(pl.col('__g033_rs')).alias('__g033_rr'), rank(pl.col('__g033_tp')).alias('__g033_rt'))
    expr = (-1.0 * low_min + delay(low_min, 5)) * pl.col('__g033_rr') * delay(pl.col('__g033_rt'), 5)
    return staged.select(expr.alias('gtja_033').cast(pl.Float64)).to_series()

def gtja_034(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #034 — 12d MA over current price ratio.

    Guotai Junan Formula
    --------------------
        MEAN(CLOSE, 12) / CLOSE

    Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    vwap = pl.col('vwap')
    return panel.select((mean(vwap, 12) / vwap).alias('gtja_034').cast(pl.Float64)).to_series()

def gtja_035(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #035 — Min of two decay-linear/EWMA-rank arms, negated.

    Guotai Junan Formula
    --------------------
        MIN(
          RANK(DECAYLINEAR(DELTA(OPEN, 1), 15)),
          RANK(DECAYLINEAR(CORR(VOLUME, OPEN*0.65 + CLOSE*0.35, 17), 7))
        ) * -1

    Daic115 substitutes DECAYLINEAR with EWMA(alpha=1/n). We follow.

    Required panel columns: ``open``, ``close``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    open_d = delta(pl.col('open'), 1)
    part1_inner = sma(open_d, 15, 1)
    weighted = pl.col('open') * 0.65 + pl.col('close') * 0.35
    cor_inner = corr(pl.col('volume'), weighted, 17)
    part2_inner = sma(cor_inner, 7, 1)
    staged = panel.with_columns(part1_inner.alias('__g035_p1'), part2_inner.alias('__g035_p2'))
    staged = staged.with_columns(rank(pl.col('__g035_p1')).alias('__g035_r1'), rank(pl.col('__g035_p2')).alias('__g035_r2'))
    expr = pl.min_horizontal(pl.col('__g035_r1'), pl.col('__g035_r2')) * -1.0
    return staged.select(expr.alias('gtja_035').cast(pl.Float64)).to_series()

def gtja_036(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #036 — Rank of 2d sum of rank-volume/rank-vwap 6d corr.

    Guotai Junan Formula
    --------------------
        RANK(SUM(CORR(RANK(VOLUME), RANK(VWAP), 6), 2))

    Required panel columns: ``volume``, ``vwap``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    staged = panel.with_columns(rank(pl.col('volume')).alias('__g036_rv'), rank(pl.col('vwap')).alias('__g036_rw'))
    staged = staged.with_columns(corr(pl.col('__g036_rv'), pl.col('__g036_rw'), 6).alias('__g036_c'))
    staged = staged.with_columns(sum_(pl.col('__g036_c'), 2).alias('__g036_s'))
    return staged.select(rank(pl.col('__g036_s')).alias('gtja_036').cast(pl.Float64)).to_series()

def gtja_037(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #037 — Negated rank of 10d acceleration of (Sum(open,5) × Sum(ret,5)).

    Guotai Junan Formula
    --------------------
        -1 * RANK((SUM(OPEN, 5) * SUM(RET, 5)) - DELAY(SUM(OPEN, 5) * SUM(RET, 5), 10))

    Daic115 uses ``ret = vwap/delay(vwap)-1`` not `returns` column —
    these differ by adj_factor + vwap vs close. We use the panel
    ``returns`` column for consistency.

    Required panel columns: ``open``, ``returns``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    vwap = pl.col('vwap')
    ret = vwap / delay(vwap, 1) - 1.0
    product = sum_(pl.col('open'), 5) * sum_(ret, 5)
    accel = product - delay(product, 10)
    staged = panel.with_columns(accel.alias('__g037_a'))
    return staged.select((-1.0 * rank(pl.col('__g037_a'))).alias('gtja_037').cast(pl.Float64)).to_series()

def gtja_038(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #038 — Conditional negated 2d high delta when high>20d-MA.

    Guotai Junan Formula
    --------------------
        (MEAN(HIGH, 20) < HIGH) ? (-1 * DELTA(HIGH, 2)) : 0

    Required panel columns: ``high``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    cond = mean(pl.col('high'), 20) < pl.col('high')
    expr = pl.when(cond).then(-1.0 * delta(pl.col('high'), 2)).otherwise(0.0)
    return panel.select(expr.alias('gtja_038').cast(pl.Float64)).to_series()

def gtja_039(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #039 — Difference of two rank(decay-linear) arms, negated.

    Guotai Junan Formula
    --------------------
        (RANK(DECAYLINEAR(DELTA(CLOSE, 2), 8)) -
         RANK(DECAYLINEAR(CORR(VWAP*0.3 + OPEN*0.7,
                              SUM(MEAN(VOLUME, 180), 37), 14), 12))) * -1

    Required panel columns: ``close``, ``vwap``, ``open``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    p1_inner = decay_linear(delta(pl.col('close'), 2), 8)
    weighted = pl.col('vwap') * 0.3 + pl.col('open') * 0.7
    vol_long = sum_(mean(pl.col('volume'), 180), 37)
    cor = corr(weighted, vol_long, 14)
    p2_inner = decay_linear(cor, 12)
    staged = panel.with_columns(p1_inner.alias('__g039_p1'), p2_inner.alias('__g039_p2'))
    staged = staged.with_columns(rank(pl.col('__g039_p1')).alias('__g039_r1'), rank(pl.col('__g039_p2')).alias('__g039_r2'))
    expr = (pl.col('__g039_r1') - pl.col('__g039_r2')) * -1.0
    return staged.select(expr.alias('gtja_039').cast(pl.Float64)).to_series()

def gtja_040(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #040 — 26d up/down volume ratio × 100.

    Guotai Junan Formula
    --------------------
        SUM((C > DELAY(C, 1) ? VOLUME : 0), 26) /
        SUM((C <= DELAY(C, 1) ? VOLUME : 0), 26) * 100

    Required panel columns: ``vwap``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    vwap = pl.col('vwap')
    cond = vwap > delay(vwap, 1)
    up = ifelse(cond, pl.col('volume'), 0.0)
    dn = ifelse(~cond, pl.col('volume'), 0.0)
    expr = sum_(up, 26) / sum_(dn, 26) * 100.0
    return panel.select(expr.alias('gtja_040').cast(pl.Float64)).to_series()
_DOC_BASE = 'docs/factor_library/gtja191'
_REF_BASE = "Guotai Junan 2017, '191 Alphas', via Daic115/alpha191 (formula only)"
_ENTRIES: list[FactorEntry] = [FactorEntry(id='gtja_021', impl=gtja_021, direction='reverse', category='momentum', description='Rolling 6d slope of MEAN(close,6) vs sequence (regbeta vs row-index)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_021.md', quality_flag=1), FactorEntry(id='gtja_022', impl=gtja_022, direction='reverse', category='mean_reversion', description='EWMA(12,1) of (close mean-detrend - 3d-lag) over 6d window', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_022.md'), FactorEntry(id='gtja_023', impl=gtja_023, direction='normal', category='volatility', description='Up-day STD share over 20d (×100), smoothed via SMA(20,1)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_023.md'), FactorEntry(id='gtja_024', impl=gtja_024, direction='normal', category='momentum', description='EWMA(5,1) of 5d price change', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_024.md'), FactorEntry(id='gtja_025', impl=gtja_025, direction='reverse', category='momentum', description='-Rank(close7d-delta) × Rank(EWMA volume ratio) × (1+Rank(150d ret sum))', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_025.md'), FactorEntry(id='gtja_026', impl=gtja_026, direction='normal', category='mean_reversion', description='(MEAN(close,12) - close) + 200d corr(vwap, delay(close,5))', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_026.md'), FactorEntry(id='gtja_027', impl=gtja_027, direction='normal', category='momentum', description='EWMA(12,1) of 3d+6d % momentum sum (Daic115 SMA-substituted-WMA)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_027.md'), FactorEntry(id='gtja_028', impl=gtja_028, direction='normal', category='momentum', description='KDJ-style 9d stochastic with two SMA(3,1) layers', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_028.md'), FactorEntry(id='gtja_029', impl=gtja_029, direction='normal', category='volume_price', description='6d % change × log(volume) (Daic115 log-vol substitution)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_029.md'), FactorEntry(id='gtja_030', impl=gtja_030, direction='normal', category='volatility', description='STUB — Fama-French residual^2 WMA (Daic115 unfinished)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_030.md', quality_flag=2), FactorEntry(id='gtja_031', impl=gtja_031, direction='reverse', category='mean_reversion', description='12d MA-distance ratio × 100', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_031.md'), FactorEntry(id='gtja_032', impl=gtja_032, direction='reverse', category='volume_price', description='-SUM(rank(corr(rank-H, rank-Vol, 3)), 3)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_032.md'), FactorEntry(id='gtja_033', impl=gtja_033, direction='normal', category='volume_price', description='Low-min change × ret-spread rank × turnover-proxy rank (amount/cap)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_033.md', quality_flag=1), FactorEntry(id='gtja_034', impl=gtja_034, direction='reverse', category='mean_reversion', description='MEAN(close, 12) / close', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_034.md'), FactorEntry(id='gtja_035', impl=gtja_035, direction='reverse', category='volume_price', description='Min of two rank(decay) arms (open delta + vol-weighted-price corr) negated', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_035.md'), FactorEntry(id='gtja_036', impl=gtja_036, direction='normal', category='volume_price', description='Rank of 2d sum of corr(rank-volume, rank-vwap, 6)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_036.md'), FactorEntry(id='gtja_037', impl=gtja_037, direction='reverse', category='momentum', description='-Rank of 10d acceleration of (sum(open,5) × sum(ret,5))', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_037.md'), FactorEntry(id='gtja_038', impl=gtja_038, direction='reverse', category='mean_reversion', description='Conditional negated 2d high-delta when high>MEAN(high,20)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_038.md'), FactorEntry(id='gtja_039', impl=gtja_039, direction='reverse', category='momentum', description='-(rank-decay close-delta - rank-decay long-vol corr)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_039.md'), FactorEntry(id='gtja_040', impl=gtja_040, direction='normal', category='volume_price', description='26d up-volume / down-volume ratio × 100', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_040.md')]
for _e in _ENTRIES:
    register_gtja191(_e)

# ---- third_party/aurumq_gtja191/aurumq_rl/factors/gtja191/batch_041_060.py ----
"""GTJA-191 factor library — batch 041 through 060.

Translated from Daic115/alpha191 (formula reference, no code vendored).
"""
import polars as pl

def gtja_041(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #041 — Negated rank of 5d-max of 3d VWAP delta.

    Guotai Junan Formula
    --------------------
        RANK(MAX(DELTA(VWAP, 3), 5)) * -1

    Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    inner = ts_max(delta(pl.col('vwap'), 3), 5)
    staged = panel.with_columns(inner.alias('__g041_x'))
    return staged.select((rank(pl.col('__g041_x')) * -1.0).alias('gtja_041').cast(pl.Float64)).to_series()

def gtja_042(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #042 — Negated rank-of-std times rolling H-V correlation.

    Guotai Junan Formula
    --------------------
        -1 * RANK(STD(HIGH, 10)) * CORR(HIGH, VOLUME, 10)

    Required panel columns: ``high``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged = panel.with_columns(std_(pl.col('high'), 10).alias('__g042_s'), corr(pl.col('high'), pl.col('volume'), 10).alias('__g042_c'))
    return staged.select((-1.0 * rank(pl.col('__g042_s')) * pl.col('__g042_c')).alias('gtja_042').cast(pl.Float64)).to_series()

def gtja_043(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #043 — 6d signed-volume sum (vwap-anchored direction).

    Guotai Junan Formula
    --------------------
        SUM((C > DELAY(C,1) ? V : (C < DELAY(C,1) ? -V : 0)), 6)

    Required panel columns: ``vwap``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    vwap = pl.col('vwap')
    cond = vwap > delay(vwap, 1)
    signed_vol = ifelse(cond, pl.col('volume'), -pl.col('volume'))
    return panel.select(sum_(signed_vol, 6).alias('gtja_043').cast(pl.Float64)).to_series()

def gtja_044(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #044 — Sum of two TSRANK(DECAYLINEAR(...)) arms.

    Guotai Junan Formula
    --------------------
        TSRANK(DECAYLINEAR(CORR(LOW, MEAN(VOLUME, 10), 7), 6), 4) +
        TSRANK(DECAYLINEAR(DELTA(VWAP, 3), 10), 15)

    Required panel columns: ``low``, ``volume``, ``vwap``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    cor_inner = corr(pl.col('low'), mean(pl.col('volume'), 10), 7)
    arm1 = ts_rank(decay_linear(cor_inner, 6), 4)
    arm2 = ts_rank(decay_linear(delta(pl.col('vwap'), 3), 10), 15)
    return panel.select((arm1 + arm2).alias('gtja_044').cast(pl.Float64)).to_series()

def gtja_045(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #045 — Rank(C0.6+O0.4 delta) × Rank(corr(VWAP, MEAN(V,150), 15)).

    Guotai Junan Formula
    --------------------
        RANK(DELTA(C*0.6 + O*0.4, 1)) * RANK(CORR(VWAP, MEAN(VOLUME, 150), 15))

    Daic115 multiplies by raw corr (not rank); we follow.

    Required panel columns: ``close``, ``open``, ``vwap``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    weighted = pl.col('close') * 0.6 + pl.col('open') * 0.4
    inner_d = delta(weighted, 1)
    corr_v = corr(pl.col('vwap'), mean(pl.col('volume'), 150), 15)
    staged = panel.with_columns(inner_d.alias('__g045_d'))
    staged = staged.with_columns(rank(pl.col('__g045_d')).alias('__g045_r'))
    return staged.select((pl.col('__g045_r') * corr_v).alias('gtja_045').cast(pl.Float64)).to_series()

def gtja_046(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #046 — Multi-window MA average / close.

    Guotai Junan Formula
    --------------------
        (MEAN(C,3) + MEAN(C,6) + MEAN(C,12) + MEAN(C,24)) / (4 * C)

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    c = pl.col('close')
    expr = (mean(c, 3) + mean(c, 6) + mean(c, 12) + mean(c, 24)) / (4.0 * c)
    return panel.select(expr.alias('gtja_046').cast(pl.Float64)).to_series()

def gtja_047(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #047 — Smoothed RSV: SMA((TSMAX(H,6)-C)/(TSMAX(H,6)-TSMIN(L,6))*100, 9, 1).

    Required panel columns: ``high``, ``low``, ``close``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    hmax = ts_max(pl.col('high'), 6)
    lmin = ts_min(pl.col('low'), 6)
    raw = (hmax - pl.col('close')) / (hmax - lmin) * 100.0
    return panel.select(sma(raw, 9, 1).alias('gtja_047').cast(pl.Float64)).to_series()

def gtja_048(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #048 — Rank(sign-sum) × SUM(V,5)/SUM(V,20).

    Guotai Junan Formula
    --------------------
        -1 * RANK(SIGN(C - DELAY(C,1)) + SIGN(DELAY(C,1) - DELAY(C,2)) +
                  SIGN(DELAY(C,2) - DELAY(C,3))) * SUM(V,5) / SUM(V,20)

    Required panel columns: ``close``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    c = pl.col('close')
    s1 = sign_(delta(c, 1))
    s2 = sign_(delta(delay(c, 1), 1))
    s3 = sign_(delta(delay(c, 2), 1))
    sgn_sum = s1 + s2 + s3
    staged = panel.with_columns(sgn_sum.alias('__g048_s'))
    staged = staged.with_columns(rank(pl.col('__g048_s')).alias('__g048_r'))
    expr = pl.col('__g048_r') * sum_(pl.col('volume'), 5) / sum_(pl.col('volume'), 20)
    return staged.select(expr.alias('gtja_048').cast(pl.Float64)).to_series()

def gtja_049(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #049 — Down-day range share over 12d.

    Guotai Junan Formula
    --------------------
        cond = (H + L) >= (DELAY(H, 1) + DELAY(L, 1))
        part = MAX(|H - DELAY(H, 1)|, |L - DELAY(L, 1)|)
        SUM((!cond ? part : 0), 12) /
        (SUM((!cond ? part : 0), 12) + SUM((cond ? part : 0), 12))

    Required panel columns: ``high``, ``low``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volatility``
    """
    h = pl.col('high')
    lw = pl.col('low')
    cond = h + lw >= delay(h, 1) + delay(lw, 1)
    part = pl.max_horizontal(abs_(h - delay(h, 1)), abs_(lw - delay(lw, 1)))
    s_dn = sumif(part, 12, ~cond)
    s_up = sumif(part, 12, cond)
    return panel.select((s_dn / (s_dn + s_up)).alias('gtja_049').cast(pl.Float64)).to_series()

def gtja_050(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #050 — Down-share minus Up-share asymmetric range ratio.

    Guotai Junan Formula
    --------------------
        cond1 = (H + L) <= (DELAY(H,1) + DELAY(L,1))
        cond2 = (H + L) >= (DELAY(H,1) + DELAY(L,1))
        part = MAX(|H - DELAY(H,1)|, |L - DELAY(L,1)|)
        SUM(!cond1?part:0, 12) / (SUM(!cond1?part:0, 12) + SUM(!cond2?part:0, 12)) -
        SUM(!cond2?part:0, 12) / (SUM(!cond2?part:0, 12) + SUM(!cond1?part:0, 12))

    Required panel columns: ``high``, ``low``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volatility``
    """
    h = pl.col('high')
    lw = pl.col('low')
    cond1 = h + lw <= delay(h, 1) + delay(lw, 1)
    cond2 = h + lw >= delay(h, 1) + delay(lw, 1)
    part = pl.max_horizontal(abs_(h - delay(h, 1)), abs_(lw - delay(lw, 1)))
    s_a = sumif(part, 12, ~cond1)
    s_b = sumif(part, 12, ~cond2)
    expr = (s_a - s_b) / (s_a + s_b + 1e-07)
    return panel.select(expr.alias('gtja_050').cast(pl.Float64)).to_series()

def gtja_051(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #051 — Down-share asymmetric range ratio over 12d.

    Guotai Junan Formula
    --------------------
        cond1 = (H + L) <= (DELAY(H,1) + DELAY(L,1))
        cond2 = (H + L) >= (DELAY(H,1) + DELAY(L,1))
        part = MAX(|H - DELAY(H,1)|, |L - DELAY(L,1)|)
        SUM(!cond1?part:0, 12) / (SUM(!cond1?part:0, 12) + SUM(!cond2?part:0, 12))

    Required panel columns: ``high``, ``low``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volatility``
    """
    h = pl.col('high')
    lw = pl.col('low')
    cond1 = h + lw <= delay(h, 1) + delay(lw, 1)
    cond2 = h + lw >= delay(h, 1) + delay(lw, 1)
    part = pl.max_horizontal(abs_(h - delay(h, 1)), abs_(lw - delay(lw, 1)))
    s_a = sumif(part, 12, ~cond1)
    s_b = sumif(part, 12, ~cond2)
    expr = s_a / (s_a + s_b + 1e-07)
    return panel.select(expr.alias('gtja_051').cast(pl.Float64)).to_series()

def gtja_052(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #052 — 26d upward / downward typical-price pressure × 100.

    Guotai Junan Formula
    --------------------
        SUM(MAX(0, H - DELAY((H+L+C)/3, 1)), 26) /
        SUM(MAX(0, DELAY((H+L+C)/3, 1) - L), 26) * 100

    Required panel columns: ``high``, ``low``, ``close``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    typ = (pl.col('high') + pl.col('low') + pl.col('close')) / 3.0
    typ_lag = delay(typ, 1)
    up = pl.max_horizontal(
        (pl.col('high') - typ_lag).cast(pl.Float64),
        pl.lit(0.0, dtype=pl.Float64),
    )
    dn = pl.max_horizontal(
        (typ_lag - pl.col('low')).cast(pl.Float64),
        pl.lit(0.0, dtype=pl.Float64),
    )
    expr = sum_(up, 26) / sum_(dn, 26) * 100.0
    return panel.select(expr.alias('gtja_052').cast(pl.Float64)).to_series()

def gtja_053(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #053 — % of up-days over 12d × 100.

    Guotai Junan Formula
    --------------------
        COUNT(CLOSE > DELAY(CLOSE, 1), 12) / 12 * 100

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    c = pl.col('close')
    cond = (c > delay(c, 1)).cast(pl.Float64)
    return panel.select((sum_(cond, 12) / 12.0 * 100.0).alias('gtja_053').cast(pl.Float64)).to_series()

def gtja_054(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #054 — Negated rank of std-of-asymmetric-spread + close-open corr.

    Guotai Junan Formula
    --------------------
        -1 * RANK(STD(|C-O| + (C-O), 10) + CORR(C, O, 10))

    Required panel columns: ``close``, ``open``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volatility``
    """
    diff = pl.col('close') - pl.col('open')
    inner = std_(abs_(diff) + diff, 10) + corr(pl.col('close'), pl.col('open'), 10)
    staged = panel.with_columns(inner.alias('__g054_x'))
    return staged.select((-1.0 * rank(pl.col('__g054_x'))).alias('gtja_054').cast(pl.Float64)).to_series()

def gtja_055(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #055 — 20d sum of TR-normalised acceleration × max(|H-C-1|, |L-C-1|).

    Guotai Junan Formula
    --------------------
        SUM(16 * (C - DELAY(C,1) + (C-O)/2 + DELAY(C,1) - DELAY(O,1)) /
            (asymmetric TR normaliser per spec) *
            MAX(|H - DELAY(C,1)|, |L - DELAY(C,1)|), 20)

    Required panel columns: ``open``, ``high``, ``low``, ``close``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    c = pl.col('close')
    o = pl.col('open')
    h = pl.col('high')
    lw = pl.col('low')
    p1 = abs_(h - delay(c, 1))
    p2 = abs_(lw - delay(c, 1))
    p3 = abs_(h - delay(lw, 1))
    p4 = abs_(delay(c, 1) - delay(o, 1))
    var1 = p1 + p2 / 2.0 + p4 / 4.0
    var2 = p2 + p1 / 2.0 + p4 / 4.0
    var3 = p3 + p4 / 4.0
    cond_a = (p1 > p2) & (p1 > p3)
    cond_b = (p2 > p3) & (p2 > p1)
    denom = pl.when(cond_a).then(var1).otherwise(pl.when(cond_b).then(var2).otherwise(var3))
    accel = c - delay(c, 1) + (c - o) / 2.0 + delay(c, 1) - delay(o, 1)
    inner = 16.0 * accel / denom * pl.max_horizontal(p1, p2)
    return panel.select(sum_(inner, 20).alias('gtja_055').cast(pl.Float64)).to_series()

def gtja_056(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #056 — Cross-sectional rank inequality: open-min vs corr^5.

    Guotai Junan Formula
    --------------------
        RANK(OPEN - TSMIN(OPEN, 12)) <
        RANK(RANK(CORR(SUM((H+L)/2, 19), SUM(MEAN(V, 40), 19), 13))^5)

    Returns a 0/1 indicator (cast to float).

    Required panel columns: ``open``, ``high``, ``low``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    o = pl.col('open')
    arm1_inner = o - ts_min(o, 12)
    sum_mid = sum_((pl.col('high') + pl.col('low')) / 2.0, 19)
    sum_v = sum_(mean(pl.col('volume'), 40), 19)
    cor = corr(sum_mid, sum_v, 13)
    staged = panel.with_columns(arm1_inner.alias('__g056_a1'), cor.alias('__g056_c'))
    staged = staged.with_columns(rank(pl.col('__g056_a1')).alias('__g056_r1'), rank(pl.col('__g056_c')).alias('__g056_rc'))
    staged = staged.with_columns(rank(pl.col('__g056_rc') ** 5.0).alias('__g056_r2'))
    return staged.select((pl.col('__g056_r1') < pl.col('__g056_r2')).cast(pl.Float64).alias('gtja_056')).to_series()

def gtja_057(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #057 — 3-period EWMA of 9-period stochastic %K.

    Guotai Junan Formula
    --------------------
        SMA((C - TSMIN(L, 9)) / (TSMAX(H, 9) - TSMIN(L, 9)) * 100, 3, 1)

    Required panel columns: ``close``, ``low``, ``high``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    raw = (pl.col('close') - ts_min(pl.col('low'), 9)) / (ts_max(pl.col('high'), 9) - ts_min(pl.col('low'), 9)) * 100.0
    return panel.select(sma(raw, 3, 1).alias('gtja_057').cast(pl.Float64)).to_series()

def gtja_058(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #058 — % of up-days over 20d × 100 (vwap-anchored).

    Guotai Junan Formula
    --------------------
        COUNT(CLOSE > DELAY(CLOSE, 1), 20) / 20 * 100

    Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    v = pl.col('vwap')
    cond = (v > delay(v, 1)).cast(pl.Float64)
    return panel.select((sum_(cond, 20) / 20.0 * 100.0).alias('gtja_058').cast(pl.Float64)).to_series()

def gtja_059(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #059 — 20d sum of close-vs-extreme conditional flow (vwap).

    Guotai Junan Formula
    --------------------
        SUM((C = DELAY(C,1) ? 0 : C - (C > DELAY(C,1) ?
             MIN(L, DELAY(C,1)) : MAX(H, DELAY(C,1)))), 20)

    Required panel columns: ``vwap``, ``low``, ``high``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    v = pl.col('vwap')
    v_lag = delay(v, 1)
    pivot = pl.when(v > v_lag).then(pl.min_horizontal(pl.col('low'), v_lag)).otherwise(pl.max_horizontal(pl.col('high'), v_lag))
    inner = pl.when(v != v_lag).then(v - pivot).otherwise(0.0)
    return panel.select(sum_(inner, 20).alias('gtja_059').cast(pl.Float64)).to_series()

def gtja_060(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #060 — 20d sum of normalised mid-range × volume.

    Guotai Junan Formula
    --------------------
        SUM(((C - L) - (H - C)) / (H - L) * VOLUME, 20)

    Required panel columns: ``close``, ``low``, ``high``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    base = (pl.col('close') - pl.col('low') - (pl.col('high') - pl.col('close'))) / (pl.col('high') - pl.col('low'))
    return panel.select(sum_(base * pl.col('volume'), 20).alias('gtja_060').cast(pl.Float64)).to_series()
_DOC_BASE = 'docs/factor_library/gtja191'
_REF_BASE = "Guotai Junan 2017, '191 Alphas', via Daic115/alpha191 (formula only)"
_ENTRIES: list[FactorEntry] = [FactorEntry(id='gtja_041', impl=gtja_041, direction='reverse', category='momentum', description='-Rank(MAX(DELTA(VWAP, 3), 5))', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_041.md'), FactorEntry(id='gtja_042', impl=gtja_042, direction='reverse', category='volume_price', description='-RANK(STD(H, 10)) × CORR(H, V, 10)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_042.md'), FactorEntry(id='gtja_043', impl=gtja_043, direction='normal', category='volume_price', description='6d signed-volume sum (vwap-anchored direction)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_043.md'), FactorEntry(id='gtja_044', impl=gtja_044, direction='normal', category='volume_price', description='TS-rank decay corr(low, MA(V,10), 7) + TS-rank decay delta(VWAP, 3)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_044.md'), FactorEntry(id='gtja_045', impl=gtja_045, direction='reverse', category='volume_price', description='Rank(C0.6+O0.4 delta) × CORR(VWAP, MEAN(V, 150), 15)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_045.md'), FactorEntry(id='gtja_046', impl=gtja_046, direction='reverse', category='mean_reversion', description='(MA3 + MA6 + MA12 + MA24) / (4 × close)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_046.md'), FactorEntry(id='gtja_047', impl=gtja_047, direction='reverse', category='mean_reversion', description='Smoothed inverse-RSV: SMA((TSMAX(H,6)-C)/(TSMAX(H,6)-TSMIN(L,6))×100, 9, 1)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_047.md'), FactorEntry(id='gtja_048', impl=gtja_048, direction='reverse', category='momentum', description='-Rank(3-day sign sum) × SUM(V,5)/SUM(V,20)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_048.md'), FactorEntry(id='gtja_049', impl=gtja_049, direction='reverse', category='volatility', description='Down-day range share over 12d', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_049.md'), FactorEntry(id='gtja_050', impl=gtja_050, direction='reverse', category='volatility', description='Down-share - Up-share asymmetric range ratio over 12d', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_050.md', quality_flag=1), FactorEntry(id='gtja_051', impl=gtja_051, direction='reverse', category='volatility', description='Down-share asymmetric range ratio over 12d', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_051.md', quality_flag=1), FactorEntry(id='gtja_052', impl=gtja_052, direction='normal', category='momentum', description='26d typical-price up/down pressure ratio × 100', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_052.md'), FactorEntry(id='gtja_053', impl=gtja_053, direction='normal', category='momentum', description='% of up-days over 12d × 100', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_053.md'), FactorEntry(id='gtja_054', impl=gtja_054, direction='reverse', category='volatility', description='-Rank(STD(|C-O|+(C-O), 10) + CORR(C, O, 10))', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_054.md'), FactorEntry(id='gtja_055', impl=gtja_055, direction='normal', category='momentum', description='20d sum of TR-normalised acceleration × max(|H-C-1|,|L-C-1|)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_055.md', quality_flag=1), FactorEntry(id='gtja_056', impl=gtja_056, direction='reverse', category='volume_price', description='Rank-inequality: open-min vs rank-corr^5 (returns 0/1)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_056.md'), FactorEntry(id='gtja_057', impl=gtja_057, direction='normal', category='momentum', description='3-period EWMA of 9-period stochastic %K', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_057.md'), FactorEntry(id='gtja_058', impl=gtja_058, direction='normal', category='momentum', description='% of up-days over 20d × 100 (vwap-anchored)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_058.md'), FactorEntry(id='gtja_059', impl=gtja_059, direction='reverse', category='volume_price', description='20d sum of close-vs-extreme conditional flow (vwap-anchored)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_059.md'), FactorEntry(id='gtja_060', impl=gtja_060, direction='normal', category='volume_price', description='20d sum of normalised mid-range × volume', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_060.md')]
for _e in _ENTRIES:
    register_gtja191(_e)

# ---- third_party/aurumq_gtja191/aurumq_rl/factors/gtja191/batch_061_080.py ----
"""GTJA-191 factor library — batch 061 through 080.

Translated from Daic115/alpha191 (formula reference, no code vendored).
"""
import polars as pl

def gtja_061(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #061 — Max of rank(decay-VWAP-delta), rank(decay-rank-corr).

    Guotai Junan Formula
    --------------------
        MAX(RANK(DECAYLINEAR(DELTA(VWAP, 1), 12)),
            RANK(DECAYLINEAR(RANK(CORR(LOW, MEAN(V, 80), 8)), 17))) * -1

    Daic115 omits the `* -1`. We follow Daic115 (no negation).

    Required panel columns: ``vwap``, ``low``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    arm1_inner = decay_linear(delta(pl.col('vwap'), 1), 12)
    cor = corr(pl.col('low'), mean(pl.col('volume'), 80), 8)
    staged = panel.with_columns(arm1_inner.alias('__g061_a1'), cor.alias('__g061_c'))
    staged = staged.with_columns(rank(pl.col('__g061_a1')).alias('__g061_r1'), rank(pl.col('__g061_c')).alias('__g061_rc'))
    staged = staged.with_columns(decay_linear(pl.col('__g061_rc'), 17).alias('__g061_a2_inner'))
    staged = staged.with_columns(rank(pl.col('__g061_a2_inner')).alias('__g061_r2'))
    return staged.select(pl.max_horizontal(pl.col('__g061_r1'), pl.col('__g061_r2')).alias('gtja_061').cast(pl.Float64)).to_series()

def gtja_062(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #062 — -CORR(HIGH, RANK(TURN), 5) — turnover proxied by amount/cap.

    Guotai Junan Formula
    --------------------
        -CORR(HIGH, RANK(TURN), 5)

    Daic115 references ``data["turn"]`` (turnover_rate) which we don't
    have in the synthetic panel. Use ``amount / cap`` as turnover proxy.
    Reference parquet does NOT include gtja_062; reference test is
    skipped.

    Required panel columns: ``high``, ``amount``, ``cap``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    turn_proxy = pl.col('amount') / pl.col('cap')
    staged = panel.with_columns(rank(turn_proxy).alias('__g062_rt'))
    return staged.select((-1.0 * corr(pl.col('high'), pl.col('__g062_rt'), 5)).alias('gtja_062').cast(pl.Float64)).to_series()

def gtja_063(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #063 — 6-day RSI: SMA(MAX(C-C-1, 0), 6, 1) / SMA(|C-C-1|, 6, 1) × 100.

    Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    v = pl.col('vwap')
    diff = v - delay(v, 1)
    up = pl.max_horizontal(
        diff.cast(pl.Float64), pl.lit(0.0, dtype=pl.Float64)
    )
    return panel.select((sma(up, 6, 1) / sma(abs_(diff), 6, 1) * 100.0).alias('gtja_063').cast(pl.Float64)).to_series()

def gtja_064(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #064 — Max of two rank(decay-corr) arms.

    Guotai Junan Formula
    --------------------
        MAX(
          RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 4), 4)),
          RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE), RANK(MEAN(V, 60)), 4), 13), 14))
        ) * -1

    Daic115 omits the `* -1`; we follow.

    Required panel columns: ``vwap``, ``volume``, ``close``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged = panel.with_columns(rank(pl.col('vwap')).alias('__g064_rw'), rank(pl.col('volume')).alias('__g064_rv'), rank(pl.col('close')).alias('__g064_rc'), rank(mean(pl.col('volume'), 60)).alias('__g064_rmv'))
    staged = staged.with_columns(corr(pl.col('__g064_rw'), pl.col('__g064_rv'), 4).alias('__g064_c1'), corr(pl.col('__g064_rc'), pl.col('__g064_rmv'), 4).alias('__g064_c2'))
    staged = staged.with_columns(decay_linear(pl.col('__g064_c1'), 4).alias('__g064_a1_inner'), ts_max(pl.col('__g064_c2'), 13).alias('__g064_a2_max'))
    staged = staged.with_columns(decay_linear(pl.col('__g064_a2_max'), 14).alias('__g064_a2_inner'))
    staged = staged.with_columns(rank(pl.col('__g064_a1_inner')).alias('__g064_r1'), rank(pl.col('__g064_a2_inner')).alias('__g064_r2'))
    return staged.select(pl.max_horizontal(pl.col('__g064_r1'), pl.col('__g064_r2')).alias('gtja_064').cast(pl.Float64)).to_series()

def gtja_065(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #065 — MEAN(close, 6) / close.

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    c = pl.col('close')
    return panel.select((mean(c, 6) / c).alias('gtja_065').cast(pl.Float64)).to_series()

def gtja_066(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #066 — (close - MA6) / MA6 × 100.

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    c = pl.col('close')
    m6 = mean(c, 6)
    return panel.select(((c - m6) / m6 * 100.0).alias('gtja_066').cast(pl.Float64)).to_series()

def gtja_067(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #067 — 24-day RSI.

    Guotai Junan Formula
    --------------------
        SMA(MAX(C-C-1, 0), 24, 1) / SMA(|C-C-1|, 24, 1) × 100

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    c = pl.col('close')
    diff = c - delay(c, 1)
    up = pl.max_horizontal(
        diff.cast(pl.Float64), pl.lit(0.0, dtype=pl.Float64)
    )
    return panel.select((sma(up, 24, 1) / sma(abs_(diff), 24, 1) * 100.0).alias('gtja_067').cast(pl.Float64)).to_series()

def gtja_068(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #068 — EWMA(15,2) of mid-price acceleration × (H-L)/V.

    Guotai Junan Formula
    --------------------
        SMA(((H+L)/2 - (DELAY(H,1)+DELAY(L,1))/2) * (H-L)/V, 15, 2)

    Required panel columns: ``high``, ``low``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    mid = (pl.col('high') + pl.col('low')) / 2.0
    mid_lag = (delay(pl.col('high'), 1) + delay(pl.col('low'), 1)) / 2.0
    inner = (mid - mid_lag) * (pl.col('high') - pl.col('low')) / pl.col('volume')
    return panel.select(sma(inner, 15, 2).alias('gtja_068').cast(pl.Float64)).to_series()

def gtja_069(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #069 — DTM/DBM 20-day asymmetric momentum ratio.

    Guotai Junan Formula
    --------------------
        DTM = (O <= DELAY(O,1)) ? 0 : MAX((H-O), (O-DELAY(O,1)))
        DBM = (O >= DELAY(O,1)) ? 0 : MAX((O-L), (O-DELAY(O,1)))
        S_DTM = SUM(DTM, 20); S_DBM = SUM(DBM, 20)
        S_DTM > S_DBM ? (S_DTM - S_DBM)/S_DTM
                       : S_DTM == S_DBM ? 0 : (S_DTM - S_DBM)/S_DBM

    Required panel columns: ``open``, ``high``, ``low``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    o = pl.col('open')
    o_lag = delay(o, 1)
    dtm_inner = pl.max_horizontal(pl.col('high') - o, o - o_lag)
    dbm_inner = pl.max_horizontal(o - pl.col('low'), o - o_lag)
    dtm = pl.when(o <= o_lag).then(0.0).otherwise(dtm_inner)
    dbm = pl.when(o >= o_lag).then(0.0).otherwise(dbm_inner)
    s_dtm = sum_(dtm, 20)
    s_dbm = sum_(dbm, 20)
    expr = pl.when(s_dtm > s_dbm).then((s_dtm - s_dbm) / s_dtm).otherwise(pl.when(s_dtm == s_dbm).then(0.0).otherwise((s_dtm - s_dbm) / s_dbm))
    return panel.select(expr.alias('gtja_069').cast(pl.Float64)).to_series()

def gtja_070(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #070 — 6-day std of amount.

    Required panel columns: ``amount``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volatility``
    """
    return panel.select(std_(pl.col('amount'), 6).alias('gtja_070').cast(pl.Float64)).to_series()

def gtja_071(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #071 — (close - MA24) / MA24 × 100.

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    c = pl.col('close')
    m = mean(c, 24)
    return panel.select(((c - m) / m * 100.0).alias('gtja_071').cast(pl.Float64)).to_series()

def gtja_072(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #072 — SMA((TSMAX(H,6)-C)/(TSMAX(H,6)-TSMIN(L,6)) × 100, 15, 1).

    Required panel columns: ``high``, ``low``, ``close``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    hmax = ts_max(pl.col('high'), 6)
    lmin = ts_min(pl.col('low'), 6)
    raw = (hmax - pl.col('close')) / (hmax - lmin) * 100.0
    return panel.select(sma(raw, 15, 1).alias('gtja_072').cast(pl.Float64)).to_series()

def gtja_073(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #073 — -TS_RANK(decay-decay-corr) - RANK(decay-corr-MA30).

    Guotai Junan Formula
    --------------------
        -1 * TS_RANK(DECAYLINEAR(DECAYLINEAR(CORR(C, V, 10), 16), 4), 5) -
        RANK(DECAYLINEAR(CORR(VWAP, MEAN(V, 30), 4), 3))

    Required panel columns: ``close``, ``volume``, ``vwap``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    c1 = corr(pl.col('close'), pl.col('volume'), 10)
    arm1_inner = decay_linear(decay_linear(c1, 16), 4)
    arm1 = -1.0 * ts_rank(arm1_inner, 5)
    c2 = corr(pl.col('vwap'), mean(pl.col('volume'), 30), 4)
    arm2_inner = decay_linear(c2, 3)
    staged = panel.with_columns(arm2_inner.alias('__g073_a2_inner'))
    staged = staged.with_columns(rank(pl.col('__g073_a2_inner')).alias('__g073_r2'))
    return staged.select((arm1 - pl.col('__g073_r2')).alias('gtja_073').cast(pl.Float64)).to_series()

def gtja_074(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #074 — Rank(corr(sum-weighted, sum-MA-V, 7)) + rank(corr(rank-VWAP, rank-V, 6)).

    Guotai Junan Formula
    --------------------
        RANK(CORR(SUM(L*0.35 + VWAP*0.65, 20), SUM(MEAN(V, 40), 20), 7)) +
        RANK(CORR(RANK(VWAP), RANK(VOLUME), 6))

    Required panel columns: ``low``, ``vwap``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    weighted = pl.col('low') * 0.35 + pl.col('vwap') * 0.65
    arm1_corr = corr(sum_(weighted, 20), sum_(mean(pl.col('volume'), 40), 20), 7)
    staged = panel.with_columns(arm1_corr.alias('__g074_c1'), rank(pl.col('vwap')).alias('__g074_rw'), rank(pl.col('volume')).alias('__g074_rv'))
    staged = staged.with_columns(corr(pl.col('__g074_rw'), pl.col('__g074_rv'), 6).alias('__g074_c2'))
    staged = staged.with_columns(rank(pl.col('__g074_c1')).alias('__g074_r1'), rank(pl.col('__g074_c2')).alias('__g074_r2'))
    return staged.select((pl.col('__g074_r1') + pl.col('__g074_r2')).alias('gtja_074').cast(pl.Float64)).to_series()

def gtja_075(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #075 — Conditional up-day count ratio vs benchmark down-days.

    Guotai Junan Formula
    --------------------
        COUNT(C > O & BENCH_C < BENCH_O, 50) / COUNT(BENCH_C < BENCH_O, 50)

    Daic115 substitutes a CS-mean return for the benchmark. We do the
    same: ``bench_ret = mean(returns)`` per trade_date, then `bench<0`
    is our "benchmark down" indicator.

    Required panel columns: ``close``, ``open``, ``returns``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    cs_ret = pl.col('returns').mean().over('trade_date')
    bench_dn = cs_ret < 0.0
    cond_a = (pl.col('close') > pl.col('open')) & bench_dn
    cond_b = (pl.col('close') != pl.col('open')) & bench_dn
    a = sum_(cond_a.cast(pl.Float64), 50)
    b = sum_(cond_b.cast(pl.Float64), 50)
    return panel.select((a / b).alias('gtja_075').cast(pl.Float64)).to_series()

def gtja_076(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #076 — STD(|ret|/V, 20) / MEAN(|ret|/V, 20).

    Required panel columns: ``close``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volatility``
    """
    c = pl.col('close')
    rel_ret_per_v = abs_(c / delay(c, 1) - 1.0) / pl.col('volume')
    return panel.select((std_(rel_ret_per_v, 20) / mean(rel_ret_per_v, 20)).alias('gtja_076').cast(pl.Float64)).to_series()

def gtja_077(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #077 — MIN of two rank(DECAYLINEAR(...)) arms.

    Guotai Junan Formula
    --------------------
        MIN(
          RANK(DECAYLINEAR(((H+L)/2 + H) - (VWAP + H), 20)),
          RANK(DECAYLINEAR(CORR((H+L)/2, MEAN(V, 40), 3), 6))
        )

    Required panel columns: ``high``, ``low``, ``vwap``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    mid = (pl.col('high') + pl.col('low')) / 2.0
    inner1 = mid + pl.col('high') - (pl.col('vwap') + pl.col('high'))
    arm1_inner = decay_linear(inner1, 20)
    cor = corr(mid, mean(pl.col('volume'), 40), 3)
    arm2_inner = decay_linear(cor, 6)
    staged = panel.with_columns(arm1_inner.alias('__g077_a1'), arm2_inner.alias('__g077_a2'))
    staged = staged.with_columns(rank(pl.col('__g077_a1')).alias('__g077_r1'), rank(pl.col('__g077_a2')).alias('__g077_r2'))
    return staged.select(pl.min_horizontal(pl.col('__g077_r1'), pl.col('__g077_r2')).alias('gtja_077').cast(pl.Float64)).to_series()

def gtja_078(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #078 — CCI-style typical price oscillator.

    Guotai Junan Formula
    --------------------
        ((H+L+C)/3 - MA((H+L+C)/3, 12)) /
        (0.015 * MEAN(|C - MEAN((H+L+C)/3, 12)|, 12))

    Required panel columns: ``high``, ``low``, ``close``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``mean_reversion``
    """
    typ = (pl.col('high') + pl.col('low') + pl.col('close')) / 3.0
    typ_ma = mean(typ, 12)
    expr = (typ - typ_ma) / (0.015 * mean(abs_(pl.col('close') - typ_ma), 12))
    return panel.select(expr.alias('gtja_078').cast(pl.Float64)).to_series()

def gtja_079(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #079 — 12-day RSI.

    Guotai Junan Formula
    --------------------
        SMA(MAX(C-C-1, 0), 12, 1) / SMA(|C-C-1|, 12, 1) × 100

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    c = pl.col('close')
    diff = c - delay(c, 1)
    up = pl.max_horizontal(
        diff.cast(pl.Float64), pl.lit(0.0, dtype=pl.Float64)
    )
    return panel.select((sma(up, 12, 1) / sma(abs_(diff), 12, 1) * 100.0).alias('gtja_079').cast(pl.Float64)).to_series()

def gtja_080(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #080 — (V - DELAY(V, 5)) / DELAY(V, 5) × 100.

    Required panel columns: ``volume``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    v = pl.col('volume')
    v_lag = delay(v, 5)
    return panel.select(((v - v_lag) / v_lag * 100.0).alias('gtja_080').cast(pl.Float64)).to_series()
_DOC_BASE = 'docs/factor_library/gtja191'
_REF_BASE = "Guotai Junan 2017, '191 Alphas', via Daic115/alpha191 (formula only)"
_ENTRIES: list[FactorEntry] = [FactorEntry(id='gtja_061', impl=gtja_061, direction='reverse', category='volume_price', description='MAX of rank(decay-VWAP-delta), rank(decay-rank(corr(L, MA(V,80), 8)))', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_061.md'), FactorEntry(id='gtja_062', impl=gtja_062, direction='reverse', category='volume_price', description='-CORR(HIGH, RANK(turn proxy=amount/cap), 5)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_062.md', quality_flag=1), FactorEntry(id='gtja_063', impl=gtja_063, direction='normal', category='momentum', description='6-day RSI (vwap-anchored)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_063.md'), FactorEntry(id='gtja_064', impl=gtja_064, direction='reverse', category='volume_price', description='MAX of two rank(decay-corr) arms (vwap-vol, close-MA-vol)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_064.md'), FactorEntry(id='gtja_065', impl=gtja_065, direction='reverse', category='mean_reversion', description='MEAN(close, 6) / close', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_065.md'), FactorEntry(id='gtja_066', impl=gtja_066, direction='reverse', category='mean_reversion', description='(close - MA6) / MA6 × 100', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_066.md'), FactorEntry(id='gtja_067', impl=gtja_067, direction='normal', category='momentum', description='24-day RSI', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_067.md'), FactorEntry(id='gtja_068', impl=gtja_068, direction='normal', category='volume_price', description='EWMA(15,2) of mid-price acceleration × (H-L)/V', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_068.md'), FactorEntry(id='gtja_069', impl=gtja_069, direction='normal', category='momentum', description='DTM/DBM 20-day asymmetric momentum ratio', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_069.md', quality_flag=1), FactorEntry(id='gtja_070', impl=gtja_070, direction='normal', category='volatility', description='6-day std of amount', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_070.md'), FactorEntry(id='gtja_071', impl=gtja_071, direction='reverse', category='mean_reversion', description='(close - MA24) / MA24 × 100', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_071.md'), FactorEntry(id='gtja_072', impl=gtja_072, direction='reverse', category='mean_reversion', description='SMA((TSMAX(H,6)-C)/(TSMAX(H,6)-TSMIN(L,6)) × 100, 15, 1)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_072.md'), FactorEntry(id='gtja_073', impl=gtja_073, direction='reverse', category='volume_price', description='-TS_RANK(decay-decay-corr(C,V)) - RANK(decay-corr(VWAP, MA30(V)))', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_073.md', quality_flag=1), FactorEntry(id='gtja_074', impl=gtja_074, direction='normal', category='volume_price', description='Rank-corr-sum-weighted-prices + Rank-corr-rank(VWAP, V)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_074.md'), FactorEntry(id='gtja_075', impl=gtja_075, direction='normal', category='momentum', description='Up-day count ratio vs CS-mean-return-as-benchmark down-days (50d)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_075.md'), FactorEntry(id='gtja_076', impl=gtja_076, direction='reverse', category='volatility', description='STD(|ret|/V, 20) / MEAN(|ret|/V, 20)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_076.md'), FactorEntry(id='gtja_077', impl=gtja_077, direction='reverse', category='volume_price', description='MIN of two rank(decay) arms (synthetic-mid-vs-VWAP, mid-MA40V-corr)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_077.md'), FactorEntry(id='gtja_078', impl=gtja_078, direction='normal', category='mean_reversion', description='CCI-style typical-price oscillator (12d)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_078.md'), FactorEntry(id='gtja_079', impl=gtja_079, direction='normal', category='momentum', description='12-day RSI', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_079.md'), FactorEntry(id='gtja_080', impl=gtja_080, direction='normal', category='volume_price', description='(V - DELAY(V, 5)) / DELAY(V, 5) × 100', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_080.md')]
for _e in _ENTRIES:
    register_gtja191(_e)

# ---- third_party/aurumq_gtja191/aurumq_rl/factors/gtja191/batch_081_100.py ----
"""GTJA-191 factor library — batch 081 through 100.

Translated from Daic115/alpha191 (formula reference, no code vendored).
"""
import polars as pl

def gtja_081(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #081 — EWMA(21, 2) of volume.

    Required panel columns: ``volume``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    return panel.select(sma(pl.col('volume'), 21, 2).alias('gtja_081').cast(pl.Float64)).to_series()

def gtja_082(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #082 — SMA((TSMAX(H,6)-C)/(TSMAX(H,6)-TSMIN(L,6))*100, 20, 1).

    Required panel columns: ``high``, ``low``, ``close``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    hmax = ts_max(pl.col('high'), 6)
    lmin = ts_min(pl.col('low'), 6)
    raw = (hmax - pl.col('close')) / (hmax - lmin) * 100.0
    return panel.select(sma(raw, 20, 1).alias('gtja_082').cast(pl.Float64)).to_series()

def gtja_083(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #083 — Negated rank of 5d covariance of rank(H) vs rank(V).

    Guotai Junan Formula
    --------------------
        -1 * RANK(COVARIANCE(RANK(HIGH), RANK(VOLUME), 5))

    Required panel columns: ``high``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged = panel.with_columns(rank(pl.col('high')).alias('__g083_rh'), rank(pl.col('volume')).alias('__g083_rv'))
    staged = staged.with_columns(covariance(pl.col('__g083_rh'), pl.col('__g083_rv'), 5).alias('__g083_c'))
    return staged.select((-1.0 * rank(pl.col('__g083_c'))).alias('gtja_083').cast(pl.Float64)).to_series()

def gtja_084(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #084 — 20d signed-volume sum (close-direction).

    Guotai Junan Formula
    --------------------
        SUM((C > DELAY(C,1) ? V : (C < DELAY(C,1) ? -V : 0)), 20)

    Required panel columns: ``close``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    c = pl.col('close')
    cond_up = c > delay(c, 1)
    cond_dn = c < delay(c, 1)
    signed_vol = pl.when(cond_up).then(pl.col('volume')).otherwise(pl.when(cond_dn).then(-pl.col('volume')).otherwise(0.0))
    return panel.select(sum_(signed_vol, 20).alias('gtja_084').cast(pl.Float64)).to_series()

def gtja_085(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #085 — Volume-ratio TS-rank × negated 7d-close-delta TS-rank.

    Guotai Junan Formula
    --------------------
        TSRANK(V / MEAN(V, 20), 20) * TSRANK(-1 * DELTA(C, 7), 8)

    Required panel columns: ``close``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    vol_ratio = pl.col('volume') / mean(pl.col('volume'), 20)
    arm1 = ts_rank(vol_ratio, 20)
    arm2 = ts_rank(-1.0 * delta(pl.col('close'), 7), 8)
    return panel.select((arm1 * arm2).alias('gtja_085').cast(pl.Float64)).to_series()

def gtja_086(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #086 — 20/10/0 close-acceleration regime ternary.

    Guotai Junan Formula
    --------------------
        part1 = (DELAY(C, 20) - DELAY(C, 10)) / 10
        part2 = (DELAY(C, 10) - C) / 10
        if (0.25 < (part1 - part2)) -1
        elif ((part1 - part2) < 0) 1
        else -1 * (C - DELAY(C, 1))

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    c = pl.col('close')
    part1 = (delay(c, 20) - delay(c, 10)) / 10.0
    part2 = (delay(c, 10) - c) / 10.0
    diff = part1 - part2
    base = -1.0 * (c - delay(c, 1))
    expr = pl.when(diff > 0.25).then(-1.0).otherwise(pl.when(diff < 0.0).then(1.0).otherwise(base))
    return panel.select(expr.alias('gtja_086').cast(pl.Float64)).to_series()

def gtja_087(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #087 — Rank-decay(vwap delta) + TS-rank-decay(asymmetric spread), negated.

    Guotai Junan Formula
    --------------------
        (RANK(DECAYLINEAR(DELTA(VWAP, 4), 7)) +
         TSRANK(DECAYLINEAR(((L*0.9 + L*0.1) - VWAP) / (O - (H+L)/2), 11), 7)) * -1

    Required panel columns: ``vwap``, ``low``, ``open``, ``high``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    arm1_inner = decay_linear(delta(pl.col('vwap'), 4), 7)
    spread_num = pl.col('low') * 0.9 + pl.col('low') * 0.1 - pl.col('vwap')
    spread_den = pl.col('open') - (pl.col('high') + pl.col('low')) / 2.0 + 1e-07
    arm2_inner = decay_linear(spread_num / spread_den, 11)
    arm2 = ts_rank(arm2_inner, 7)
    staged = panel.with_columns(arm1_inner.alias('__g087_a1_inner'))
    staged = staged.with_columns(rank(pl.col('__g087_a1_inner')).alias('__g087_r1'))
    return staged.select(((pl.col('__g087_r1') + arm2) * -1.0).alias('gtja_087').cast(pl.Float64)).to_series()

def gtja_088(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #088 — 20-day % change × 100.

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    c = pl.col('close')
    return panel.select(((c - delay(c, 20)) / delay(c, 20) * 100.0).alias('gtja_088').cast(pl.Float64)).to_series()

def gtja_089(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #089 — MACD-style oscillator: 2*(SMA13 - SMA27 - SMA10(SMA13-SMA27)).

    Guotai Junan Formula
    --------------------
        2 * (SMA(C, 13, 2) - SMA(C, 27, 2) -
             SMA(SMA(C, 13, 2) - SMA(C, 27, 2), 10, 2))

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    c = pl.col('close')
    ma_short = sma(c, 13, 2)
    ma_long = sma(c, 27, 2)
    diff = ma_short - ma_long
    return panel.select((2.0 * (ma_short - ma_long - sma(diff, 10, 2))).alias('gtja_089').cast(pl.Float64)).to_series()

def gtja_090(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #090 — Negated rank of 5d corr(rank-VWAP, rank-V).

    Required panel columns: ``vwap``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged = panel.with_columns(rank(pl.col('vwap')).alias('__g090_rw'), rank(pl.col('volume')).alias('__g090_rv'))
    staged = staged.with_columns(corr(pl.col('__g090_rw'), pl.col('__g090_rv'), 5).alias('__g090_c'))
    return staged.select((-1.0 * rank(pl.col('__g090_c'))).alias('gtja_090').cast(pl.Float64)).to_series()

def gtja_091(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #091 — Negated product of two rank arms.

    Guotai Junan Formula
    --------------------
        -1 * RANK(C - TSMAX(C, 5)) * RANK(CORR(MEAN(V, 40), L, 5))

    Required panel columns: ``close``, ``volume``, ``low``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    c = pl.col('close')
    arm1_inner = c - ts_max(c, 5)
    cor = corr(mean(pl.col('volume'), 40), pl.col('low'), 5)
    staged = panel.with_columns(arm1_inner.alias('__g091_a1'), cor.alias('__g091_c'))
    staged = staged.with_columns(rank(pl.col('__g091_a1')).alias('__g091_r1'), rank(pl.col('__g091_c')).alias('__g091_r2'))
    return staged.select((pl.col('__g091_r1') * pl.col('__g091_r2') * -1.0).alias('gtja_091').cast(pl.Float64)).to_series()

def gtja_092(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #092 — MAX of rank-decay-delta and TS-rank-decay-abs-corr, negated.

    Guotai Junan Formula
    --------------------
        MAX(
          RANK(DECAYLINEAR(DELTA(C*0.35 + VWAP*0.65, 2), 3)),
          TSRANK(DECAYLINEAR(|CORR(MEAN(V, 180), C, 13)|, 5), 15)
        ) * -1

    Required panel columns: ``close``, ``vwap``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    weighted = pl.col('close') * 0.35 + pl.col('vwap') * 0.65
    arm1_inner = decay_linear(delta(weighted, 2), 3)
    cor = corr(mean(pl.col('volume'), 180), pl.col('close'), 13)
    arm2_inner = decay_linear(abs_(cor), 5)
    arm2 = ts_rank(arm2_inner, 15)
    staged = panel.with_columns(arm1_inner.alias('__g092_a1'))
    staged = staged.with_columns(rank(pl.col('__g092_a1')).alias('__g092_r1'))
    return staged.select((pl.max_horizontal(pl.col('__g092_r1'), arm2) * -1.0).alias('gtja_092').cast(pl.Float64)).to_series()

def gtja_093(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #093 — 20d sum of conditional max(O-L, O-DELAY(O,1)) when O<DELAY(O,1).

    Guotai Junan Formula
    --------------------
        SUM((O >= DELAY(O, 1) ? 0 : MAX(O - L, O - DELAY(O, 1))), 20)

    Required panel columns: ``open``, ``low``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volatility``
    """
    o = pl.col('open')
    o_lag = delay(o, 1)
    inner = pl.max_horizontal(o - pl.col('low'), o - o_lag)
    expr = pl.when(o >= o_lag).then(0.0).otherwise(inner)
    return panel.select(sum_(expr, 20).alias('gtja_093').cast(pl.Float64)).to_series()

def gtja_094(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #094 — 30d signed-volume sum.

    Guotai Junan Formula
    --------------------
        SUM((C > DELAY(C, 1) ? V : (C < DELAY(C, 1) ? -V : 0)), 30)

    Required panel columns: ``close``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    c = pl.col('close')
    cond_up = c > delay(c, 1)
    cond_dn = c < delay(c, 1)
    signed_vol = pl.when(cond_up).then(pl.col('volume')).otherwise(pl.when(cond_dn).then(-pl.col('volume')).otherwise(0.0))
    return panel.select(sum_(signed_vol, 30).alias('gtja_094').cast(pl.Float64)).to_series()

def gtja_095(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #095 — 20-day std of amount.

    Required panel columns: ``amount``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volatility``
    """
    return panel.select(std_(pl.col('amount'), 20).alias('gtja_095').cast(pl.Float64)).to_series()

def gtja_096(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #096 — SMA(SMA(stochastic-%K, 3, 1), 3, 1).

    Guotai Junan Formula
    --------------------
        SMA(SMA((C - TSMIN(L, 9)) / (TSMAX(H, 9) - TSMIN(L, 9)) * 100, 3, 1), 3, 1)

    Required panel columns: ``close``, ``low``, ``high``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    raw = (pl.col('close') - ts_min(pl.col('low'), 9)) / (ts_max(pl.col('high'), 9) - ts_min(pl.col('low'), 9)) * 100.0
    return panel.select(sma(sma(raw, 3, 1), 3, 1).alias('gtja_096').cast(pl.Float64)).to_series()

def gtja_097(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #097 — 10-day std of volume.

    Required panel columns: ``volume``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volatility``
    """
    return panel.select(std_(pl.col('volume'), 10).alias('gtja_097').cast(pl.Float64)).to_series()

def gtja_098(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #098 — Long-trend ternary: 100d MA acceleration regime.

    Guotai Junan Formula
    --------------------
        cond = DELTA(SUM(C, 100)/100, 100) / DELAY(C, 100)
        cond <= 0.05 ? -1*(C - TSMIN(C, 100)) : -1 * DELTA(C, 3)

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    c = pl.col('close')
    cond_val = delta(sum_(c, 100) / 100.0, 100) / delay(c, 100)
    branch_low = -1.0 * (c - ts_min(c, 100))
    branch_hi = -1.0 * delta(c, 3)
    expr = pl.when(cond_val <= 0.05).then(branch_low).otherwise(branch_hi)
    return panel.select(expr.alias('gtja_098').cast(pl.Float64)).to_series()

def gtja_099(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #099 — Negated rank of 5d covariance of rank(C) vs rank(V).

    Required panel columns: ``close``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged = panel.with_columns(rank(pl.col('close')).alias('__g099_rc'), rank(pl.col('volume')).alias('__g099_rv'))
    staged = staged.with_columns(covariance(pl.col('__g099_rc'), pl.col('__g099_rv'), 5).alias('__g099_co'))
    return staged.select((-1.0 * rank(pl.col('__g099_co'))).alias('gtja_099').cast(pl.Float64)).to_series()

def gtja_100(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #100 — 20-day std of volume.

    Required panel columns: ``volume``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volatility``
    """
    return panel.select(std_(pl.col('volume'), 20).alias('gtja_100').cast(pl.Float64)).to_series()
_DOC_BASE = 'docs/factor_library/gtja191'
_REF_BASE = "Guotai Junan 2017, '191 Alphas', via Daic115/alpha191 (formula only)"
_ENTRIES: list[FactorEntry] = [FactorEntry(id='gtja_081', impl=gtja_081, direction='normal', category='volume_price', description='EWMA(21, 2) of volume', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_081.md'), FactorEntry(id='gtja_082', impl=gtja_082, direction='reverse', category='mean_reversion', description='SMA((TSMAX(H,6)-C)/(TSMAX(H,6)-TSMIN(L,6))×100, 20, 1)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_082.md'), FactorEntry(id='gtja_083', impl=gtja_083, direction='reverse', category='volume_price', description='-RANK(COV(RANK(H), RANK(V), 5))', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_083.md'), FactorEntry(id='gtja_084', impl=gtja_084, direction='normal', category='volume_price', description='20d signed-volume sum (close-direction)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_084.md'), FactorEntry(id='gtja_085', impl=gtja_085, direction='reverse', category='momentum', description='TSRANK(V/MA20V, 20) × TSRANK(-DELTA(C, 7), 8)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_085.md'), FactorEntry(id='gtja_086', impl=gtja_086, direction='reverse', category='momentum', description='20/10/0 close-acceleration regime ternary', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_086.md'), FactorEntry(id='gtja_087', impl=gtja_087, direction='reverse', category='volume_price', description='-(rank-decay-vwap-delta + TS-rank-decay-asymmetric-spread)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_087.md'), FactorEntry(id='gtja_088', impl=gtja_088, direction='normal', category='momentum', description='20-day % change × 100', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_088.md'), FactorEntry(id='gtja_089', impl=gtja_089, direction='normal', category='momentum', description='MACD-style oscillator: 2*(SMA13 - SMA27 - SMA10(SMA13-SMA27))', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_089.md'), FactorEntry(id='gtja_090', impl=gtja_090, direction='reverse', category='volume_price', description='-RANK(CORR(RANK(VWAP), RANK(V), 5))', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_090.md'), FactorEntry(id='gtja_091', impl=gtja_091, direction='reverse', category='volume_price', description='-RANK(C - TSMAX(C, 5)) × RANK(CORR(MEAN(V, 40), L, 5))', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_091.md'), FactorEntry(id='gtja_092', impl=gtja_092, direction='reverse', category='volume_price', description='-MAX of rank-decay-delta + TS-rank-decay-|corr| arms', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_092.md'), FactorEntry(id='gtja_093', impl=gtja_093, direction='normal', category='volatility', description='20d sum of conditional max(O-L, O-O-1) when O<O-1', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_093.md'), FactorEntry(id='gtja_094', impl=gtja_094, direction='normal', category='volume_price', description='30d signed-volume sum', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_094.md'), FactorEntry(id='gtja_095', impl=gtja_095, direction='normal', category='volatility', description='20-day std of amount', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_095.md'), FactorEntry(id='gtja_096', impl=gtja_096, direction='normal', category='momentum', description='Double-smoothed stochastic %K (3,1)(3,1)', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_096.md'), FactorEntry(id='gtja_097', impl=gtja_097, direction='normal', category='volatility', description='10-day std of volume', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_097.md'), FactorEntry(id='gtja_098', impl=gtja_098, direction='reverse', category='momentum', description='100d MA-acceleration regime ternary', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_098.md'), FactorEntry(id='gtja_099', impl=gtja_099, direction='reverse', category='volume_price', description='-RANK(COV(RANK(C), RANK(V), 5))', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_099.md'), FactorEntry(id='gtja_100', impl=gtja_100, direction='normal', category='volatility', description='20-day std of volume', references=(_REF_BASE,), formula_doc_path=f'{_DOC_BASE}/gtja_100.md')]
for _e in _ENTRIES:
    register_gtja191(_e)

# ---- third_party/aurumq_gtja191/aurumq_rl/factors/gtja191/batch_101_120.py ----
"""GTJA-191 batch 101..120 (20 factors).

Each factor is implemented as a polars callable that takes the enriched
panel ``pl.DataFrame`` and returns a ``pl.Series`` aligned to the panel
rows. Self-registers into :data:`GTJA191_REGISTRY` at import time.

The Guotai Junan Alpha-191 formulas are sourced from the public report
(国泰君安证券 191 短周期价量因子, 2017). Numerical reference: Daic115/alpha191
(no LICENSE — we use it as a *formula reference only* and never vendor
its code; only the resulting reference parquet is committed).

Special factor in this batch
----------------------------

* ``gtja_116`` — Daic115's reference uses qlib's ``rolling_slope``.
  We implement it natively using :func:`regbeta` against
  :func:`sequence` (the GTJA paper formula is
  ``REGBETA(CLOSE, SEQUENCE, 20)``). Quality flag = 0.
"""
import polars as pl

def _bool_lt_signed(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """``(a < b) * -1`` — Daic115 returns -1/0 with null where either is null."""
    return pl.when(a.is_null() | b.is_null()).then(None).otherwise((a < b).cast(pl.Float64) * -1.0).cast(pl.Float64)

def gtja_101(panel: pl.DataFrame) -> pl.Series:
    """GTJA #101 — VWAP-volume corr ranked < volume-mean corr ranked.

    Guotai Junan Formula
    --------------------
        ((RANK(CORR(CLOSE, SUM(MEAN(VOLUME, 30), 37), 15)) <
          RANK(CORR(RANK(((HIGH * 0.1) + (VWAP * 0.9))),
                    RANK(VOLUME), 11))) * -1)

    Polars Implementation Notes
    ---------------------------
    Two-stage ``with_columns`` to materialise the per-stock CORRs before
    cross-section ranking, since polars cannot mix CS+TS partitions in a
    single expression.

    Direction: ``reverse`` (binary -1/0 — multiply by -1 in spec).
    Category: ``correlation``.
    """
    df = panel.with_columns([mean(pl.col('volume'), 30).alias('__mv30'), rank(pl.col('high') * 0.1 + pl.col('vwap') * 0.9).alias('__rp'), rank(pl.col('volume')).alias('__rv')])
    df = df.with_columns([sum_(pl.col('__mv30'), 37).alias('__smv30_37')])
    df = df.with_columns([corr(pl.col('close'), pl.col('__smv30_37'), 15).alias('__c1'), corr(pl.col('__rp'), pl.col('__rv'), 11).alias('__c2')])
    df = df.with_columns([rank(pl.col('__c1')).alias('__r1'), rank(pl.col('__c2')).alias('__r2')])
    return df.select(_bool_lt_signed(pl.col('__r1'), pl.col('__r2')).alias('gtja_101')).to_series()
register_gtja191(FactorEntry(id='gtja_101', impl=gtja_101, direction='reverse', category='correlation', description='VWAP-volume corr ranked < volume-mean corr ranked', references=('Guotai Junan 191 short-period factor report, 2017',)))

def gtja_102(panel: pl.DataFrame) -> pl.Series:
    """GTJA #102 — Volume RSI: SMA(MAX(dV,0))/SMA(|dV|).

    Guotai Junan Formula
    --------------------
        SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/
        SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100
    """
    dv = pl.col('volume') - delay(pl.col('volume'), 1)
    pos = pl.when(dv.is_null()).then(None).when(dv > 0).then(dv).otherwise(0.0)
    num = sma(pos, 6, 1)
    den = sma(dv.abs(), 6, 1)
    expr = (num / den * 100.0).alias('gtja_102')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_102', impl=gtja_102, direction='normal', category='volume', description='Volume RSI — SMA(max(dV,0))/SMA(|dV|)*100'))

def gtja_103(panel: pl.DataFrame) -> pl.Series:
    """GTJA #103 — (20-LOWDAY(LOW,20))/20*100 — recency of recent low.

    Implemented via :func:`ts_min` + per-stock back-search using a
    closed-form: distance-to-min as ``20 - argmin``. We use ``regbeta``-
    style trick: Daic115 has a slow ``LOWDAY`` here. We fall back to the
    operator's slow path :func:`_ops.lowday`.
    """
    expr = ((20.0 - lowday(pl.col('low'), 20)) / 20.0 * 100.0).alias('gtja_103')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_103', impl=gtja_103, direction='normal', category='momentum', description='(20-LOWDAY(LOW,20))/20*100 — recency-of-low oscillator'))

def gtja_104(panel: pl.DataFrame) -> pl.Series:
    """GTJA #104 — -1 * DELTA(CORR(HIGH,VOL,5),5) * RANK(STD(CLOSE,20)).

    Guotai Junan Formula
    --------------------
        -1 * (DELTA(CORR(HIGH, VOLUME, 5), 5) * RANK(STD(CLOSE, 20)))
    """
    df = panel.with_columns([corr(pl.col('high'), pl.col('volume'), 5).alias('__c'), std_(pl.col('close'), 20).alias('__s')])
    df = df.with_columns([delta(pl.col('__c'), 5).alias('__dc'), rank(pl.col('__s')).alias('__rs')])
    return df.select((-1.0 * pl.col('__dc') * pl.col('__rs')).alias('gtja_104')).to_series()
register_gtja191(FactorEntry(id='gtja_104', impl=gtja_104, direction='reverse', category='correlation', description='-1 * delta(corr(high,vol,5),5) * rank(std(close,20))'))

def gtja_105(panel: pl.DataFrame) -> pl.Series:
    """GTJA #105 — -1 * CORR(RANK(OPEN), RANK(VOLUME), 10)."""
    df = panel.with_columns([rank(pl.col('open')).alias('__ro'), rank(pl.col('volume')).alias('__rv')])
    return df.select((-1.0 * corr(pl.col('__ro'), pl.col('__rv'), 10)).alias('gtja_105')).to_series()
register_gtja191(FactorEntry(id='gtja_105', impl=gtja_105, direction='reverse', category='correlation', description='-1 * corr(rank(open), rank(volume), 10)'))

def gtja_106(panel: pl.DataFrame) -> pl.Series:
    """GTJA #106 — CLOSE - DELAY(CLOSE, 20). Pure 20-day momentum."""
    expr = (pl.col('close') - delay(pl.col('close'), 20)).alias('gtja_106')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_106', impl=gtja_106, direction='normal', category='momentum', description='20-day close momentum'))

def gtja_107(panel: pl.DataFrame) -> pl.Series:
    """GTJA #107 — Triple-rank gap product across O-H/O-C/O-L.

    Guotai Junan Formula
    --------------------
        ((-1 * RANK(OPEN - DELAY(HIGH, 1))) *
          RANK(OPEN - DELAY(CLOSE, 1))) *
          RANK(OPEN - DELAY(LOW, 1))
    """
    df = panel.with_columns([(pl.col('open') - delay(pl.col('high'), 1)).alias('__a'), (pl.col('open') - delay(pl.col('close'), 1)).alias('__b'), (pl.col('open') - delay(pl.col('low'), 1)).alias('__c')])
    return df.select((-1.0 * rank(pl.col('__a')) * rank(pl.col('__b')) * rank(pl.col('__c'))).alias('gtja_107')).to_series()
register_gtja191(FactorEntry(id='gtja_107', impl=gtja_107, direction='reverse', category='momentum', description='-rank(o-prev_h)*rank(o-prev_c)*rank(o-prev_l)'))

def gtja_108(panel: pl.DataFrame) -> pl.Series:
    """GTJA #108 — RANK(HIGH - MIN(HIGH,2)) ^ RANK(CORR(VWAP, MA(V,120),6))) * -1.

    The XOR-looking caret in the paper is exponentiation in Daic115's
    pandas reference (``a ** b``); we follow that.
    """
    df = panel.with_columns([(pl.col('high') - ts_min(pl.col('high'), 2)).alias('__d'), corr(pl.col('vwap'), mean(pl.col('volume'), 120), 6).alias('__c')])
    df = df.with_columns([rank(pl.col('__d')).alias('__rd'), rank(pl.col('__c')).alias('__rc')])
    expr = (pl.col('__rd').pow(pl.col('__rc')) * -1.0).alias('gtja_108')
    return df.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_108', impl=gtja_108, direction='reverse', category='correlation', description='(rank(high-min(high,2)) ^ rank(corr(vwap, ma(v,120), 6))) * -1'))

def gtja_109(panel: pl.DataFrame) -> pl.Series:
    """GTJA #109 — SMA(H-L,10,2) / SMA(SMA(H-L,10,2),10,2).

    Range-relative trend oscillator.
    """
    hl = pl.col('high') - pl.col('low')
    inner = sma(hl, 10, 2)
    expr = (inner / sma(inner, 10, 2)).alias('gtja_109')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_109', impl=gtja_109, direction='normal', category='volatility', description='SMA(H-L,10,2) / SMA(SMA(H-L,10,2),10,2)'))

def gtja_110(panel: pl.DataFrame) -> pl.Series:
    """GTJA #110 — SUM(MAX(0,H-prev_C),20) / SUM(MAX(0,prev_C-L),20) * 100.

    Buying-pressure / selling-pressure ratio over 20 days.
    """
    pc = delay(pl.col('close'), 1)
    up = pl.when(pl.col('high') - pc > 0).then(pl.col('high') - pc).otherwise(0.0)
    dn = pl.when(pc - pl.col('low') > 0).then(pc - pl.col('low')).otherwise(0.0)
    expr = (sum_(up, 20) / sum_(dn, 20) * 100.0).alias('gtja_110')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_110', impl=gtja_110, direction='normal', category='momentum', description='20d up-move-vs-down-move pressure ratio'))

def gtja_111(panel: pl.DataFrame) -> pl.Series:
    """GTJA #111 — VOL * intra-day position SMA differential."""
    rng = pl.col('high') - pl.col('low')
    pos = (pl.col('close') - pl.col('low') - (pl.col('high') - pl.col('close'))) / rng
    weighted = pl.col('volume') * pos
    expr = (sma(weighted, 11, 2) - sma(weighted, 4, 2)).alias('gtja_111')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_111', impl=gtja_111, direction='normal', category='volume', description='Volume * intraday position SMA-differential (11-4)'))

def gtja_112(panel: pl.DataFrame) -> pl.Series:
    """GTJA #112 — Up-vs-down 12-day cumulative move ratio (CMO-style)."""
    dc = pl.col('close') - delay(pl.col('close'), 1)
    up = pl.when(dc > 0).then(dc).otherwise(0.0)
    dn = pl.when(dc < 0).then(-dc).otherwise(0.0)
    sup = sum_(up, 12)
    sdn = sum_(dn, 12)
    expr = ((sup - sdn) / (sup + sdn) * 100.0).alias('gtja_112')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_112', impl=gtja_112, direction='normal', category='momentum', description='Chande momentum oscillator over 12 days'))

def gtja_113(panel: pl.DataFrame) -> pl.Series:
    """GTJA #113 — -1 * RANK(SUM(DELAY(CLOSE,5),20)/20) * CORR(C,V,2) * RANK(CORR(SUM(C,5), SUM(C,20),2)).

    NB: Daic115 broadcasts the inner ``corr_period=20`` rather than the
    paper's ``2`` for the second corr; we follow Daic115 (parity goal).
    """
    df = panel.with_columns([(sum_(delay(pl.col('close'), 5), 20) / 20.0).alias('__a'), corr(pl.col('close'), pl.col('volume'), 20).alias('__c1'), corr(sum_(pl.col('close'), 5), sum_(pl.col('close'), 20), 20).alias('__c2')])
    df = df.with_columns([rank(pl.col('__a')).alias('__ra'), rank(pl.col('__c2')).alias('__rc2')])
    return df.select((-1.0 * pl.col('__ra') * pl.col('__c1') * pl.col('__rc2')).alias('gtja_113')).to_series()
register_gtja191(FactorEntry(id='gtja_113', impl=gtja_113, direction='reverse', category='correlation', description='-rank(sum(delay(c,5),20)/20)*corr(c,v,20)*rank(corr(sum_c5,sum_c20,20))'))

def gtja_114(panel: pl.DataFrame) -> pl.Series:
    """GTJA #114 — Range-over-MA scaled by VWAP-Close gap.

    Numerical safety
    ----------------
    Three potential div-by-zero spots, all hit on A-share limit-up days
    (一字板, high=low=close=vwap):
      1. ``(high - low) / mean(close, 5)`` — denominator on suspended /
         IPO bars can be 0
      2. ``part / (vwap - close)`` — vwap-close=0 on limit-up days
      3. outer ``(rpd*rrv) / den`` — den ≈ 0 when part = 0

    Original code had ``+ 1e-7`` only for #2, which let ~7000 inf cells
    through per year. Replaced all three with ``safe_div``.
    """
    part = safe_div(pl.col('high') - pl.col('low'), mean(pl.col('close'), 5))
    df = panel.with_columns([part.alias('__p'), delay(part, 2).alias('__pd')])
    df = df.with_columns([rank(pl.col('__pd')).alias('__rpd'), rank(rank(pl.col('volume'))).alias('__rrv')])
    den = safe_div(pl.col('__p'), pl.col('vwap') - pl.col('close'))
    df = df.with_columns(den.alias('__den'))
    expr = safe_div(pl.col('__rpd') * pl.col('__rrv'), pl.col('__den')).alias('gtja_114')
    return df.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_114', impl=gtja_114, direction='normal', category='volatility', description='rank(delay(rng/ma5,2))*rank(rank(v)) / ((rng/ma5)/(vwap-c))'))

def gtja_115(panel: pl.DataFrame) -> pl.Series:
    """GTJA #115 — Pow of two corrs: (HIGH*0.9+CLOSE*0.1)~MA(V,30) ^ HL2 mid-rank ~ vol-rank."""
    df = panel.with_columns([corr(pl.col('high') * 0.9 + pl.col('close') * 0.1, mean(pl.col('volume'), 30), 10).alias('__c1')])
    df = df.with_columns([ts_rank((pl.col('high') + pl.col('low')) / 2.0, 4).alias('__t1'), ts_rank(pl.col('volume'), 10).alias('__t2')])
    df = df.with_columns([corr(pl.col('__t1'), pl.col('__t2'), 7).alias('__c2')])
    df = df.with_columns([rank(pl.col('__c1')).alias('__r1'), rank(pl.col('__c2')).alias('__r2')])
    expr = pl.col('__r1').pow(pl.col('__r2')).alias('gtja_115')
    return df.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_115', impl=gtja_115, direction='normal', category='correlation', description='rank(corr(0.9H+0.1C, ma(v,30),10)) ^ rank(corr(tsr_HL2_4, tsr_v_10, 7))'))

def gtja_116(panel: pl.DataFrame) -> pl.Series:
    """GTJA #116 — REGBETA(CLOSE, SEQUENCE(20), 20).

    Rolling slope of CLOSE on a monotonic 1..N time index. Daic115 uses
    qlib's ``rolling_slope`` (slope on per-stock row index). We replicate
    that semantics by building a per-stock row counter (``cum_count``)
    and feeding it into our native :func:`regbeta`. Since the x-axis is
    monotonic, ``regbeta`` (cov/var) produces the same OLS slope as
    qlib's ``rolling_slope``.

    Direction: ``normal``. Quality flag: ``0``.
    """
    row_idx = pl.int_range(1, pl.len() + 1).over(TS_PART).cast(pl.Float64)
    df = panel.with_columns(row_idx.alias('__row_idx'))
    expr = regbeta(pl.col('close'), pl.col('__row_idx'), 20).alias('gtja_116')
    return df.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_116', impl=gtja_116, direction='normal', category='momentum', description='20-day rolling OLS slope of CLOSE on time-index', quality_flag=0))

def gtja_117(panel: pl.DataFrame) -> pl.Series:
    """GTJA #117 — TSRANK(VOL,32) * (1-TSRANK(C+H-L,16)) * (1-TSRANK(RET,32))."""
    ret = pl.col('close') / delay(pl.col('close'), 1) - 1.0
    chl = pl.col('close') + pl.col('high') - pl.col('low')
    expr = (ts_rank(pl.col('volume'), 32) * (1.0 - ts_rank(chl, 16)) * (1.0 - ts_rank(ret, 32))).alias('gtja_117')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_117', impl=gtja_117, direction='normal', category='momentum', description='tsrank(v,32) * (1-tsrank(c+h-l,16)) * (1-tsrank(ret,32))'))

def gtja_118(panel: pl.DataFrame) -> pl.Series:
    """GTJA #118 — SUM(H-O,20) / SUM(O-L,20) * 100.

    Open-relative range-skew over 20 days.
    """
    expr = (sum_(pl.col('high') - pl.col('open'), 20) / sum_(pl.col('open') - pl.col('low'), 20) * 100.0).alias('gtja_118')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_118', impl=gtja_118, direction='normal', category='volatility', description='20-day open-relative range skew (h-o vs o-l)'))

def gtja_119(panel: pl.DataFrame) -> pl.Series:
    """GTJA #119 — Decay-linear of corrs and tsranks, V→Liquidity.

    Guotai Junan Formula
    --------------------
        RANK(DECAYLINEAR(CORR(VWAP, SUM(MEAN(VOLUME,5),26), 5), 7)) -
        RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN), RANK(MEAN(VOLUME,15)),21), 9), 7), 8))
    """
    df = panel.with_columns([mean(pl.col('volume'), 5).alias('__mv5'), mean(pl.col('volume'), 15).alias('__mv15')])
    df = df.with_columns([sum_(pl.col('__mv5'), 26).alias('__smv5_26'), rank(pl.col('open')).alias('__ro'), rank(pl.col('__mv15')).alias('__rmv')])
    df = df.with_columns([corr(pl.col('vwap'), pl.col('__smv5_26'), 5).alias('__c1')])
    df = df.with_columns([corr(pl.col('__ro'), pl.col('__rmv'), 21).alias('__c2')])
    df = df.with_columns([ts_min(pl.col('__c2'), 9).alias('__m')])
    df = df.with_columns([ts_rank(pl.col('__m'), 7).alias('__tr')])
    df = df.with_columns([decay_linear(pl.col('__c1'), 7).alias('__dl1'), decay_linear(pl.col('__tr'), 8).alias('__dl2')])
    df = df.with_columns([rank(pl.col('__dl1')).alias('__r1'), rank(pl.col('__dl2')).alias('__r2')])
    expr = (pl.col('__r1') - pl.col('__r2')).alias('gtja_119')
    return df.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_119', impl=gtja_119, direction='normal', category='correlation', description='decay-linear of corr(vwap, sum_mean_v) minus decay-linear tsrank-min-corr(rank_o, rank_mv)'))

def gtja_120(panel: pl.DataFrame) -> pl.Series:
    """GTJA #120 — RANK(VWAP-CLOSE) / RANK(VWAP+CLOSE)."""
    df = panel.with_columns([rank(pl.col('vwap') - pl.col('close')).alias('__a'), rank(pl.col('vwap') + pl.col('close')).alias('__b')])
    return df.select((pl.col('__a') / pl.col('__b')).alias('gtja_120')).to_series()
register_gtja191(FactorEntry(id='gtja_120', impl=gtja_120, direction='normal', category='volume', description='rank(vwap-close) / rank(vwap+close)'))

# ---- third_party/aurumq_gtja191/aurumq_rl/factors/gtja191/batch_121_140.py ----
"""GTJA-191 batch 121..140 (20 factors).

Special factors in this batch
-----------------------------

* ``gtja_121`` — Errata factor per ``wpwp/Alpha-101-GTJA-191`` README
  (returns 0 in that repo). We implement best-effort using the paper
  formula directly. Quality flag = 1 (errata-conservative).

* ``gtja_131`` — Errata factor per same source. Implement best-effort.
  Quality flag = 1.

The XOR / power ambiguity (``^``) in the GTJA paper is resolved as
exponentiation (``**``) following Daic115's pandas reference.
"""
import polars as pl

def gtja_121(panel: pl.DataFrame) -> pl.Series:
    """GTJA #121 — Errata factor (best-effort implementation).

    Guotai Junan Formula
    --------------------
        (RANK((VWAP - MIN(VWAP, 12))) ^ TSRANK(CORR(TSRANK(VWAP, 20),
            TSRANK(MEAN(VOLUME, 60), 2), 18), 3)) * -1

    Listed in ``wpwp/Alpha-101-GTJA-191`` errata as ``return 0``. The
    Daic115 reference implements the paper formula literally. We follow
    Daic115 here (best-effort) and tag with quality_flag=1 so downstream
    code can mask it out if desired.

    Direction: ``reverse``. Quality flag: ``1``.
    """
    df = panel.with_columns([(pl.col('vwap') - ts_min(pl.col('vwap'), 12)).alias('__d'), ts_rank(pl.col('vwap'), 20).alias('__t1'), ts_rank(mean(pl.col('volume'), 60), 2).alias('__t2')])
    df = df.with_columns([corr(pl.col('__t1'), pl.col('__t2'), 18).alias('__c')])
    df = df.with_columns([rank(pl.col('__d')).alias('__r'), ts_rank(pl.col('__c'), 3).alias('__tr')])
    expr = (pl.col('__r').pow(pl.col('__tr')) * -1.0).alias('gtja_121')
    return df.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_121', impl=gtja_121, direction='reverse', category='correlation', description='(rank(vwap-min(vwap,12)) ^ tsrank(corr(tsr_vwap_20, tsr_mv60_2, 18), 3)) * -1', quality_flag=1))

def gtja_122(panel: pl.DataFrame) -> pl.Series:
    """GTJA #122 — Triple-SMA log-close TSI-style oscillator.

    Guotai Junan Formula
    --------------------
        (SMA^3(LOG(CLOSE),13,2) - DELAY(SMA^3(LOG(CLOSE),13,2), 1)) /
         DELAY(SMA^3(LOG(CLOSE),13,2), 1)
    """
    log_c = log_(pl.col('close'))
    triple = sma(sma(sma(log_c, 13, 2), 13, 2), 13, 2)
    df = panel.with_columns(triple.alias('__t'))
    expr = ((pl.col('__t') - delay(pl.col('__t'), 1)) / delay(pl.col('__t'), 1)).alias('gtja_122')
    return df.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_122', impl=gtja_122, direction='normal', category='momentum', description='Triple-SMA log-close 1-day delta ratio'))

def gtja_123(panel: pl.DataFrame) -> pl.Series:
    """GTJA #123 — Binary cross of two corr-ranks (Daic115 style with NaN map).

    Guotai Junan Formula
    --------------------
        (RANK(CORR(SUM((H+L)/2, 20), SUM(MEAN(V,60), 20), 9))
         < RANK(CORR(LOW, VOLUME, 6))) * -1
    """
    df = panel.with_columns([corr(sum_((pl.col('high') + pl.col('low')) / 2.0, 20), sum_(mean(pl.col('volume'), 60), 20), 9).alias('__c1'), corr(pl.col('low'), pl.col('volume'), 6).alias('__c2')])
    df = df.with_columns([rank(pl.col('__c1')).alias('__r1'), rank(pl.col('__c2')).alias('__r2')])
    expr = pl.when(pl.col('__r1').is_null() | pl.col('__r2').is_null()).then(None).otherwise((pl.col('__r1') < pl.col('__r2')).cast(pl.Float64) * -1.0).alias('gtja_123')
    return df.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_123', impl=gtja_123, direction='reverse', category='correlation', description='(rank(corr_HL2_sum_mv60) < rank(corr_low_vol)) * -1'))

def gtja_124(panel: pl.DataFrame) -> pl.Series:
    """GTJA #124 — (CLOSE - VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE,30)),2)."""
    df = panel.with_columns(ts_max(pl.col('close'), 30).alias('__tc'))
    df = df.with_columns(rank(pl.col('__tc')).alias('__r'))
    df = df.with_columns(decay_linear(pl.col('__r'), 2).alias('__dl'))
    expr = ((pl.col('close') - pl.col('vwap')) / pl.col('__dl')).alias('gtja_124')
    return df.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_124', impl=gtja_124, direction='normal', category='volume', description='(close-vwap) / decay_linear(rank(ts_max(close,30)), 2)'))

def gtja_125(panel: pl.DataFrame) -> pl.Series:
    """GTJA #125 — Decay-linear ratios on corr(VWAP, MA(V,80)) and DELTA(0.5C+0.5VWAP)."""
    df = panel.with_columns([corr(pl.col('vwap'), mean(pl.col('volume'), 80), 17).alias('__c'), delta((pl.col('close') + pl.col('vwap')) / 2.0, 3).alias('__d')])
    df = df.with_columns([decay_linear(pl.col('__c'), 20).alias('__dlc'), decay_linear(pl.col('__d'), 16).alias('__dld')])
    df = df.with_columns([rank(pl.col('__dlc')).alias('__r1'), rank(pl.col('__dld')).alias('__r2')])
    return df.select((pl.col('__r1') / pl.col('__r2')).alias('gtja_125')).to_series()
register_gtja191(FactorEntry(id='gtja_125', impl=gtja_125, direction='normal', category='correlation', description='rank(decay_linear(corr_vwap_mv80,20)) / rank(decay_linear(delta(0.5C+0.5VWAP,3),16))'))

def gtja_126(panel: pl.DataFrame) -> pl.Series:
    """GTJA #126 — Typical price (C+H+L)/3."""
    expr = ((pl.col('close') + pl.col('high') + pl.col('low')) / 3.0).alias('gtja_126')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_126', impl=gtja_126, direction='normal', category='price', description='(close + high + low) / 3 — typical price'))

def gtja_127(panel: pl.DataFrame) -> pl.Series:
    """GTJA #127 — RMS of pct-distance from 12-day rolling max."""
    tmax = ts_max(pl.col('close'), 12)
    inner = (100.0 * (pl.col('close') - tmax) / tmax).pow(2)
    expr = mean(inner, 12).pow(0.5).alias('gtja_127')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_127', impl=gtja_127, direction='normal', category='volatility', description='sqrt(mean((100*(c-max(c,12))/max(c,12))^2, 12))'))

def gtja_128(panel: pl.DataFrame) -> pl.Series:
    """GTJA #128 — Money-flow index over 14 days using typical price.

    Guotai Junan Formula
    --------------------
        100 - 100 / (1 + SUMIF(TP*V, 14, TP > prev_TP) /
                          SUMIF(TP*V, 14, TP < prev_TP))
    """
    tp = (pl.col('high') + pl.col('low') + pl.col('close')) / 3.0
    df = panel.with_columns(tp.alias('__tp'))
    df = df.with_columns(delay(pl.col('__tp'), 1).alias('__ptp'))
    null_mask = pl.col('__ptp').is_null()
    cond = pl.col('__tp') > pl.col('__ptp')
    tp_v = pl.col('__tp') * pl.col('volume')
    pos = pl.when(null_mask).then(None).when(cond).then(tp_v).otherwise(0.0)
    neg = pl.when(null_mask).then(None).when(~cond).then(tp_v).otherwise(0.0)
    df = df.with_columns([sum_(pos, 14).alias('__sp'), sum_(neg, 14).alias('__sn')])
    expr = (100.0 - 100.0 / (1.0 + pl.col('__sp') / (pl.col('__sn') + 1e-07))).alias('gtja_128')
    return df.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_128', impl=gtja_128, direction='normal', category='volume', description='14-day money flow index (MFI) on typical price'))

def gtja_129(panel: pl.DataFrame) -> pl.Series:
    """GTJA #129 — SUM(IFELSE(dC<0, |dC|, 0), 12). Down-move 12-day cumsum."""
    dc = pl.col('close') - delay(pl.col('close'), 1)
    expr = sum_(pl.when(dc < 0).then(dc.abs()).otherwise(0.0), 12).alias('gtja_129')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_129', impl=gtja_129, direction='reverse', category='momentum', description='12-day downside move cumsum'))

def gtja_130(panel: pl.DataFrame) -> pl.Series:
    """GTJA #130 — Decay-linear corr ratio: HL2~MA(V,40) over rank(VWAP)~rank(VOL)."""
    df = panel.with_columns([corr((pl.col('high') + pl.col('low')) / 2.0, mean(pl.col('volume'), 40), 9).alias('__c1'), rank(pl.col('vwap')).alias('__rv'), rank(pl.col('volume')).alias('__rvo')])
    df = df.with_columns([corr(pl.col('__rv'), pl.col('__rvo'), 7).alias('__c2')])
    df = df.with_columns([decay_linear(pl.col('__c1'), 10).alias('__dl1'), decay_linear(pl.col('__c2'), 3).alias('__dl2')])
    df = df.with_columns([rank(pl.col('__dl1')).alias('__r1'), rank(pl.col('__dl2')).alias('__r2')])
    return df.select((pl.col('__r1') / pl.col('__r2')).alias('gtja_130')).to_series()
register_gtja191(FactorEntry(id='gtja_130', impl=gtja_130, direction='normal', category='correlation', description='rank(decay_linear(corr(HL2,mv40,9),10)) / rank(decay_linear(corr(rk_vwap,rk_v,7),3))'))

def gtja_131(panel: pl.DataFrame) -> pl.Series:
    """GTJA #131 — Errata factor (best-effort implementation).

    Guotai Junan Formula
    --------------------
        RANK(DELTA(VWAP, 1)) ^ TSRANK(CORR(CLOSE, MEAN(VOLUME, 50), 18), 18)

    Listed in ``wpwp/Alpha-101-GTJA-191`` errata as ``return 0``. The
    Daic115 reference implements the paper literally. We follow Daic115
    (best-effort) and tag with quality_flag=1.

    Direction: ``normal``. Quality flag: ``1``.
    """
    df = panel.with_columns([delta(pl.col('vwap'), 1).alias('__d'), corr(pl.col('close'), mean(pl.col('volume'), 50), 18).alias('__c')])
    df = df.with_columns([rank(pl.col('__d')).alias('__r'), ts_rank(pl.col('__c'), 18).alias('__tr')])
    expr = pl.col('__r').pow(pl.col('__tr')).alias('gtja_131')
    return df.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_131', impl=gtja_131, direction='normal', category='correlation', description='rank(delta(vwap,1)) ^ tsrank(corr(close, mv50, 18), 18)', quality_flag=1))

def gtja_132(panel: pl.DataFrame) -> pl.Series:
    """GTJA #132 — MEAN(AMOUNT, 20). Average daily turnover."""
    expr = mean(pl.col('amount'), 20).alias('gtja_132')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_132', impl=gtja_132, direction='normal', category='volume', description='20-day mean amount (turnover proxy)'))

def gtja_133(panel: pl.DataFrame) -> pl.Series:
    """GTJA #133 — Recency-of-high vs recency-of-low oscillator."""
    expr = ((20.0 - highday(pl.col('high'), 20)) / 20.0 * 100.0 - (20.0 - lowday(pl.col('low'), 20)) / 20.0 * 100.0).alias('gtja_133')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_133', impl=gtja_133, direction='normal', category='momentum', description='(20-highday(high,20))/20*100 - (20-lowday(low,20))/20*100'))

def gtja_134(panel: pl.DataFrame) -> pl.Series:
    """GTJA #134 — (C-prev_C12)/prev_C12 * V — vol-weighted 12d return."""
    pc = delay(pl.col('close'), 12)
    expr = ((pl.col('close') - pc) / pc * pl.col('volume')).alias('gtja_134')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_134', impl=gtja_134, direction='normal', category='volume', description='(c-prev_c12)/prev_c12 * volume'))

def gtja_135(panel: pl.DataFrame) -> pl.Series:
    """GTJA #135 — SMA(DELAY(CLOSE/DELAY(CLOSE,20),1), 20, 1).

    The leading nulls (from DELAY(C,20) and DELAY(...,1)) must propagate
    through the SMA — :func:`_ops.sma` honours ``ignore_nulls`` so that
    nulls do not pollute the EWMA recursion.
    """
    inner = pl.col('close') / delay(pl.col('close'), 20)
    expr = sma(delay(inner, 1), 20, 1).alias('gtja_135')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_135', impl=gtja_135, direction='normal', category='momentum', description='SMA(delay(close/delay(close,20),1), 20, 1)'))

def gtja_136(panel: pl.DataFrame) -> pl.Series:
    """GTJA #136 — -RANK(DELTA(RET,3)) * CORR(OPEN, VOLUME, 10)."""
    ret = pl.col('close') / delay(pl.col('close'), 1) - 1.0
    df = panel.with_columns([delta(ret, 3).alias('__dr'), corr(pl.col('open'), pl.col('volume'), 10).alias('__c')])
    df = df.with_columns(rank(pl.col('__dr')).alias('__r'))
    return df.select((-1.0 * pl.col('__r') * pl.col('__c')).alias('gtja_136')).to_series()
register_gtja191(FactorEntry(id='gtja_136', impl=gtja_136, direction='reverse', category='momentum', description='-rank(delta(ret,3)) * corr(open, volume, 10)'))

def gtja_137(panel: pl.DataFrame) -> pl.Series:
    """GTJA #137 — Complex true-range-normalised price change.

    Daic115 reference implementation followed verbatim (with conditional
    decomposed using IFELSE chain).
    """
    pc = delay(pl.col('close'), 1)
    pl_ = delay(pl.col('low'), 1)
    po = delay(pl.col('open'), 1)
    abshc = (pl.col('high') - pc).abs()
    abslc = (pl.col('low') - pc).abs()
    absco = (pc - po).abs()
    abshl = (pl.col('high') - pl_).abs()
    num = 16.0 * (pl.col('close') - pc + (pl.col('close') - pl.col('open')) / 2.0 + pc - po)
    case1 = abshc + abslc / 2.0 + absco / 4.0
    case2 = abslc + abshc / 2.0 + absco / 4.0
    case3 = abshl + absco / 4.0
    cond1 = (abshc > abslc) & (abshc > abshl)
    cond2 = (abslc > abshl) & (abslc > abshc)
    den = pl.when(cond1).then(case1).when(cond2).then(case2).otherwise(case3) + 1e-07
    max_h = pl.when(abshc > abslc).then(abshc).otherwise(abslc)
    expr = (num / den * max_h).alias('gtja_137')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_137', impl=gtja_137, direction='normal', category='volatility', description="Wilders'-style TR-normalised close move"))

def gtja_138(panel: pl.DataFrame) -> pl.Series:
    """GTJA #138 — Decay-linear delta of (0.7L+0.3VWAP) minus tsrank-cascade.

    Guotai Junan Formula
    --------------------
        (RANK(DECAYLINEAR(DELTA(0.7L + 0.3VWAP, 3), 20)) -
         TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW, 8),
                                        TSRANK(MEAN(VOLUME,60), 17), 5), 19), 16), 7)) * -1
    """
    df = panel.with_columns([delta(pl.col('low') * 0.7 + pl.col('vwap') * 0.3, 3).alias('__d'), ts_rank(pl.col('low'), 8).alias('__tl'), ts_rank(mean(pl.col('volume'), 60), 17).alias('__tv')])
    df = df.with_columns(corr(pl.col('__tl'), pl.col('__tv'), 5).alias('__c'))
    df = df.with_columns(ts_rank(pl.col('__c'), 19).alias('__tc'))
    df = df.with_columns(decay_linear(pl.col('__tc'), 16).alias('__dlt'))
    df = df.with_columns(decay_linear(pl.col('__d'), 20).alias('__dld'))
    df = df.with_columns([rank(pl.col('__dld')).alias('__r'), ts_rank(pl.col('__dlt'), 7).alias('__tr')])
    expr = ((pl.col('__r') - pl.col('__tr')) * -1.0).alias('gtja_138')
    return df.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_138', impl=gtja_138, direction='reverse', category='correlation', description='(rank(dl(delta(0.7L+0.3VWAP,3),20)) - tsr(dl(tsr_corr_ll_mv60),16),7)) * -1'))

def gtja_139(panel: pl.DataFrame) -> pl.Series:
    """GTJA #139 — -1 * CORR(OPEN, VOLUME, 10)."""
    expr = (-1.0 * corr(pl.col('open'), pl.col('volume'), 10)).alias('gtja_139')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_139', impl=gtja_139, direction='reverse', category='correlation', description='-corr(open, volume, 10)'))

def gtja_140(panel: pl.DataFrame) -> pl.Series:
    """GTJA #140 — MIN of two decay-linear ranks."""
    df = panel.with_columns([(rank(pl.col('open')) + rank(pl.col('low')) - rank(pl.col('high')) - rank(pl.col('close'))).alias('__a'), ts_rank(pl.col('close'), 8).alias('__tc'), ts_rank(mean(pl.col('volume'), 60), 20).alias('__tv')])
    df = df.with_columns(corr(pl.col('__tc'), pl.col('__tv'), 8).alias('__c'))
    df = df.with_columns(decay_linear(pl.col('__c'), 7).alias('__dlc'))
    df = df.with_columns(decay_linear(pl.col('__a'), 8).alias('__dla'))
    df = df.with_columns([rank(pl.col('__dla')).alias('__r1'), ts_rank(pl.col('__dlc'), 3).alias('__r2')])
    expr = pl.min_horizontal([pl.col('__r1'), pl.col('__r2')]).alias('gtja_140')
    return df.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_140', impl=gtja_140, direction='normal', category='correlation', description='min(rank(dl(rank_OL_HC,8)), tsr(dl(corr(tsr_c8,tsr_mv60_20,8),7),3))'))

# ---- third_party/aurumq_gtja191/aurumq_rl/factors/gtja191/batch_141_160.py ----
"""GTJA-191 batch 141..160 (20 factors).

Special factors in this batch
-----------------------------

* ``gtja_143`` — STUB. Daic115 marks it ``unfinished=True`` (recursive
  SELF reference, paper formula has ambiguity:
  ``CLOSE > DELAY(CLOSE,1) ? (CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*SELF : SELF``).
  We return all-null. Quality flag = 2.

* ``gtja_147`` — Daic115 reference uses qlib's ``rolling_slope`` on
  ``MEAN(CLOSE, 12)``. We implement using our :func:`regbeta` against
  per-stock row index. Quality flag = 0.

* ``gtja_149`` — Benchmark factor. The Daic115 reference uses
  cross-section mean of CLOSE returns as proxy for the CSI300 benchmark
  (a known degraded data source per the alpha191 handoff doc §4.1). We
  follow the same proxy here so unit tests on the synthetic panel match
  the reference. Production wiring to true CSI300 OHLC is a Phase D
  task. Quality flag = 0 (formula correct, only data source degraded
  for testing).

* ``gtja_151`` — Errata factor per ``wpwp/Alpha-101-GTJA-191`` README.
  Implement best-effort (``SMA(CLOSE-DELAY(CLOSE,20),20,1)``).
  Quality flag = 1.
"""
import polars as pl

def gtja_141(panel: pl.DataFrame) -> pl.Series:
    """GTJA #141 — RANK(CORR(RANK(HIGH), RANK(MEAN(VOLUME,15)), 9)) * -1."""
    df = panel.with_columns([mean(pl.col('volume'), 15).alias('__mv15'), rank(pl.col('high')).alias('__rh')])
    df = df.with_columns(rank(pl.col('__mv15')).alias('__rmv'))
    df = df.with_columns(corr(pl.col('__rh'), pl.col('__rmv'), 9).alias('__c'))
    df = df.with_columns(rank(pl.col('__c')).alias('__r'))
    return df.select((pl.col('__r') * -1.0).alias('gtja_141')).to_series()
register_gtja191(FactorEntry(id='gtja_141', impl=gtja_141, direction='reverse', category='correlation', description='-rank(corr(rank(high), rank(mean(v,15)), 9))'))

def gtja_142(panel: pl.DataFrame) -> pl.Series:
    """GTJA #142 — Triple-rank acceleration product."""
    df = panel.with_columns([ts_rank(pl.col('close'), 10).alias('__tc'), delta(delta(pl.col('close'), 1), 1).alias('__d2'), ts_rank(pl.col('volume') / mean(pl.col('volume'), 20), 5).alias('__tv')])
    df = df.with_columns([rank(pl.col('__tc')).alias('__r1'), rank(pl.col('__d2')).alias('__r2'), rank(pl.col('__tv')).alias('__r3')])
    return df.select((-1.0 * pl.col('__r1') * pl.col('__r2') * pl.col('__r3')).alias('gtja_142')).to_series()
register_gtja191(FactorEntry(id='gtja_142', impl=gtja_142, direction='reverse', category='momentum', description='-rank(tsr_c10) * rank(d2_close) * rank(tsr_vrel_5)'))

def gtja_143(panel: pl.DataFrame) -> pl.Series:
    """GTJA #143 — Cumulative product of up-day returns (recursive SELF).

    Guotai Junan Formula
    --------------------
        CLOSE > DELAY(CLOSE,1) ? (CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*SELF : SELF

    The recursive ``SELF`` term makes the formula a path-dependent recursion:
    on up days the running value is *multiplied* by the up-day return, on
    other days it is carried forward. Two reasonable interpretations:

        (a) literal: SELF_t = ratio_t * SELF_{t-1}, where ratio_t is the
            up-day raw return ~ 0.02. This decays SELF toward 0 quickly and
            is economically meaningless.
        (b) compounded: SELF_t = (1 + ratio_t) * SELF_{t-1} = close_t /
            delay(close, 1)_t * SELF_{t-1}, which gives the cumulative
            return path of a "buy-and-hold-on-up-days" strategy.

    Daic115 leaves the body commented out; we adopt (b) — the economically
    meaningful interpretation that matches how related GTJA factors (#018,
    #053) compose returns. SELF starts at 1.0; non-up days carry forward
    unchanged. The result is monotone non-decreasing per stock and grows
    roughly as the cumulative product of up-day prices.

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``normal``. Quality flag: ``0``.
    """
    delayed = pl.col('close').shift(1).over(TS_PART)
    factor = pl.when(pl.col('close') > delayed).then(pl.col('close') / delayed).when(delayed.is_null()).then(None).otherwise(1.0)
    staged = panel.with_columns(factor.alias('__g143_factor'))
    staged = staged.with_columns(pl.col('__g143_factor').fill_null(1.0).cum_prod().over(TS_PART).alias('__g143_cum'))
    staged = staged.with_columns(pl.when(delayed.is_null()).then(None).otherwise(pl.col('__g143_cum')).alias('gtja_143').cast(pl.Float64))
    return staged.select('gtja_143').to_series()
register_gtja191(FactorEntry(id='gtja_143', impl=gtja_143, direction='normal', category='momentum', description='Cumulative product of (close/prev_close) on up days, 1.0 otherwise', quality_flag=0))

def gtja_144(panel: pl.DataFrame) -> pl.Series:
    """GTJA #144 — Down-day average abs-return / log-amount.

    Daic115's reference uses ``log(amount)`` rather than raw amount in
    the denominator (deviation from the paper). We follow Daic115 for
    parity.
    """
    pc = delay(pl.col('close'), 1)
    abs_ret = (pl.col('close') / pc - 1.0).abs() / log_(pl.col('amount'))
    cond = pl.col('close') < pc
    masked = pl.when(cond).then(abs_ret).otherwise(0.0)
    expr = (sum_(masked, 20) / count_(cond, 20)).alias('gtja_144')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_144', impl=gtja_144, direction='normal', category='volatility', description='20d down-day mean(|ret|/log(amount))'))

def gtja_145(panel: pl.DataFrame) -> pl.Series:
    """GTJA #145 — (MEAN(V,9) - MEAN(V,26)) / MEAN(V,12) * 100."""
    expr = ((mean(pl.col('volume'), 9) - mean(pl.col('volume'), 26)) / mean(pl.col('volume'), 12) * 100.0).alias('gtja_145')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_145', impl=gtja_145, direction='normal', category='volume', description='(mean(v,9) - mean(v,26)) / mean(v,12) * 100'))

def gtja_146(panel: pl.DataFrame) -> pl.Series:
    """GTJA #146 — Daic115 variant: mean(part_ma,20) * part_ma / SMA(part-part_ma)^2."""
    pc = delay(pl.col('close'), 1)
    part = (pl.col('close') - pc) / pc
    df = panel.with_columns(part.alias('__p'))
    df = df.with_columns(sma(pl.col('__p'), 61, 2).alias('__sp'))
    df = df.with_columns((pl.col('__p') - pl.col('__sp')).alias('__pm'))
    df = df.with_columns([mean(pl.col('__pm'), 20).alias('__m'), sma((pl.col('__p') - pl.col('__pm')).pow(2), 61, 2).alias('__s')])
    expr = (pl.col('__m') * pl.col('__pm') / pl.col('__s')).alias('gtja_146')
    return df.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_146', impl=gtja_146, direction='normal', category='momentum', description='mean(ret-sma(ret,61,2),20) * (ret-sma) / SMA((ret-(ret-sma))^2,61,2)'))

def gtja_147(panel: pl.DataFrame) -> pl.Series:
    """GTJA #147 — REGBETA(MEAN(CLOSE,12), SEQUENCE(12)).

    Daic115's reference uses qlib's ``rolling_slope`` on MEAN(CLOSE,12).
    We use our native :func:`regbeta` against a per-stock row index — a
    monotonic x-axis, which produces the same OLS slope.

    Direction: ``normal``. Quality flag: ``0``.
    """
    row_idx = pl.int_range(1, pl.len() + 1).over(TS_PART).cast(pl.Float64)
    df = panel.with_columns([mean(pl.col('close'), 12).alias('__mc'), row_idx.alias('__row_idx')])
    expr = regbeta(pl.col('__mc'), pl.col('__row_idx'), 12).alias('gtja_147')
    return df.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_147', impl=gtja_147, direction='normal', category='momentum', description='12-day rolling OLS slope of MEAN(close,12) on time-index', quality_flag=0))

def gtja_148(panel: pl.DataFrame) -> pl.Series:
    """GTJA #148 — (RANK(CORR(OPEN, SUM(MEAN(V,60),9), 6)) < RANK(OPEN-TSMIN(OPEN,14))) * -1."""
    df = panel.with_columns([corr(pl.col('open'), sum_(mean(pl.col('volume'), 60), 9), 6).alias('__c'), (pl.col('open') - ts_min(pl.col('open'), 14)).alias('__d')])
    df = df.with_columns([rank(pl.col('__c')).alias('__r1'), rank(pl.col('__d')).alias('__r2')])
    expr = pl.when(pl.col('__r1').is_null() | pl.col('__r2').is_null()).then(None).otherwise((pl.col('__r1') < pl.col('__r2')).cast(pl.Float64) * -1.0).alias('gtja_148')
    return df.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_148', impl=gtja_148, direction='reverse', category='correlation', description='(rank(corr(o, sum(mv60,9), 6)) < rank(o-tsmin(o,14))) * -1'))

def gtja_149(panel: pl.DataFrame) -> pl.Series:
    """GTJA #149 — Downside-beta vs benchmark over 252 days.

    Guotai Junan Formula
    --------------------
        REGBETA(
            FILTER(CLOSE/DELAY(CLOSE,1)-1, BMK < DELAY(BMK,1)),
            FILTER(BMK/DELAY(BMK,1)-1, BMK < DELAY(BMK,1)),
            252)

    Benchmark sourcing
    ------------------
    The Daic115 reference uses cross-section mean of CLOSE/DELAY(CLOSE,1)
    returns as proxy for CSI300 — a known degraded data source per the
    alpha191 handoff doc §4.1. We follow the same proxy here so unit
    tests on the synthetic panel match the reference parquet.

    Production wiring to the real CSI300 OHLC (available in our
    ``index_daily`` table from 2015-10 onwards) is a Phase D task. The
    formula itself is correct; only the data source is degraded.

    Daic115 also drops the ``FILTER`` step (commented out) — we match
    that and feed the full series into REGBETA, again for parity.

    Direction: ``normal``. Quality flag: ``0``.
    """
    ret = pl.col('close') / delay(pl.col('close'), 1) - 1.0
    df = panel.with_columns(ret.alias('__ret'))
    bench = pl.col('__ret').mean().over('trade_date')
    df = df.with_columns(bench.alias('__bench'))
    expr = regbeta(pl.col('__ret'), pl.col('__bench'), 252).alias('gtja_149')
    return df.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_149', impl=gtja_149, direction='normal', category='benchmark', description='252d beta vs cross-section-mean-return benchmark proxy (CSI300 wiring is Phase D)', quality_flag=0))

def gtja_150(panel: pl.DataFrame) -> pl.Series:
    """GTJA #150 — (C+H+L)/3 * LOG(VOLUME). Daic115 uses log(volume)."""
    expr = ((pl.col('close') + pl.col('high') + pl.col('low')) / 3.0 * log_(pl.col('volume'))).alias('gtja_150')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_150', impl=gtja_150, direction='normal', category='volume', description='typical price * log(volume) (Daic115 variant)'))

def gtja_151(panel: pl.DataFrame) -> pl.Series:
    """GTJA #151 — Errata factor (best-effort).

    Guotai Junan Formula
    --------------------
        SMA(CLOSE - DELAY(CLOSE, 20), 20, 1)

    Listed in ``wpwp/Alpha-101-GTJA-191`` errata. The formula itself is
    well-defined; we implement it as Daic115 does. Quality flag = 1
    pending downstream verification.

    Direction: ``normal``. Quality flag: ``1``.
    """
    expr = sma(pl.col('close') - delay(pl.col('close'), 20), 20, 1).alias('gtja_151')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_151', impl=gtja_151, direction='normal', category='momentum', description='SMA(close - delay(close,20), 20, 1)', quality_flag=1))

def gtja_152(panel: pl.DataFrame) -> pl.Series:
    """GTJA #152 — DEA-style triple-EWMA differential."""
    inner = sma(delay(pl.col('close') / delay(pl.col('close'), 9), 1), 9, 1)
    part = delay(inner, 1)
    expr = sma(mean(part, 12) - mean(part, 26), 9, 1).alias('gtja_152')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_152', impl=gtja_152, direction='normal', category='momentum', description='sma(mean(part,12)-mean(part,26),9,1) where part is delayed sma(c/c9,9,1)'))

def gtja_153(panel: pl.DataFrame) -> pl.Series:
    """GTJA #153 — (MA3+MA6+MA12+MA24)/4 — multi-MA average."""
    expr = ((mean(pl.col('close'), 3) + mean(pl.col('close'), 6) + mean(pl.col('close'), 12) + mean(pl.col('close'), 24)) / 4.0).alias('gtja_153')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_153', impl=gtja_153, direction='normal', category='momentum', description='Average of 4 different moving averages (BBI)'))

def gtja_154(panel: pl.DataFrame) -> pl.Series:
    """GTJA #154 — Sign indicator: -1/0 of (vwap-min(vwap,16)) < CORR(vwap, MA(V,180), 18)."""
    df = panel.with_columns([(pl.col('vwap') - ts_min(pl.col('vwap'), 16)).alias('__a'), corr(pl.col('vwap'), mean(pl.col('volume'), 180), 18).alias('__c')])
    expr = pl.when(pl.col('__a').is_null() | pl.col('__c').is_null()).then(None).otherwise(pl.when(pl.col('__a') < pl.col('__c')).then(1.0).otherwise(-1.0)).alias('gtja_154')
    return df.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_154', impl=gtja_154, direction='normal', category='correlation', description='sign((vwap-min(vwap,16)) < corr(vwap, mv180, 18))'))

def gtja_155(panel: pl.DataFrame) -> pl.Series:
    """GTJA #155 — MACD-style on volume."""
    macd = sma(pl.col('volume'), 13, 2) - sma(pl.col('volume'), 27, 2)
    signal = sma(macd, 10, 2)
    expr = (macd - signal).alias('gtja_155')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_155', impl=gtja_155, direction='normal', category='volume', description='Volume MACD (13,27,10) histogram'))

def gtja_156(panel: pl.DataFrame) -> pl.Series:
    """GTJA #156 — MAX of two decay-linear ranks * -1."""
    a = pl.col('vwap') - delay(pl.col('vwap'), 5)
    b_inner = pl.col('open') * 0.15 + pl.col('low') * 0.85
    b = -delta(b_inner, 2) / b_inner
    df = panel.with_columns([decay_linear(a, 3).alias('__dla'), decay_linear(b, 3).alias('__dlb')])
    df = df.with_columns([rank(pl.col('__dla')).alias('__r1'), rank(pl.col('__dlb')).alias('__r2')])
    expr = (pl.max_horizontal([pl.col('__r1'), pl.col('__r2')]) * -1.0).alias('gtja_156')
    return df.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_156', impl=gtja_156, direction='reverse', category='momentum', description='-max(rank(dl(d_vwap_5,3)), rank(dl(-d_inner_2/inner,3)))'))

def gtja_157(panel: pl.DataFrame) -> pl.Series:
    """GTJA #157 — TS_MIN of triple-rank-log + tsrank of delay(-ret,6)."""
    df = panel.with_columns([delta(pl.col('close') - 1.0, 5).alias('__d'), (pl.col('close') / delay(pl.col('close'), 1) - 1.0).alias('__ret')])
    df = df.with_columns([rank(-1.0 * rank(pl.col('__d'))).alias('__rd')])
    df = df.with_columns(rank(pl.col('__rd')).alias('__rrd'))
    df = df.with_columns(ts_min(pl.col('__rrd'), 2).alias('__tm'))
    df = df.with_columns(sum_(pl.col('__tm'), 1).alias('__sum'))
    df = df.with_columns(rank(rank(log_(pl.col('__sum')))).alias('__lhs'))
    df = df.with_columns([ts_min(pl.col('__lhs'), 5).alias('__lhs_min'), ts_rank(delay(-1.0 * pl.col('__ret'), 6), 5).alias('__rhs')])
    expr = (pl.col('__lhs_min') + pl.col('__rhs')).alias('gtja_157')
    return df.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_157', impl=gtja_157, direction='normal', category='momentum', description='ts_min(rank(rank(log(sum(ts_min(rank(rank(-rank(d(c-1,5)))),2),1)))),5) + tsr(delay(-ret,6),5)'))

def gtja_158(panel: pl.DataFrame) -> pl.Series:
    """GTJA #158 — ((H - SMA(C,15,2)) - (L - SMA(C,15,2))) / C — high-low spread / close."""
    s = sma(pl.col('close'), 15, 2)
    expr = ((pl.col('high') - s - (pl.col('low') - s)) / pl.col('close')).alias('gtja_158')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_158', impl=gtja_158, direction='normal', category='volatility', description='(H - L) / C — high-low spread normalised by close (SMA terms cancel)'))

def gtja_159(panel: pl.DataFrame) -> pl.Series:
    """GTJA #159 — Three-window cumulative range-position oscillator."""
    pc = delay(pl.col('close'), 1)
    p2 = pl.min_horizontal([pl.col('low'), pc])
    p3 = pl.max_horizontal([pl.col('high'), pc])
    p1 = p3 - p2
    df = panel.with_columns([p1.alias('__p1'), p2.alias('__p2'), p3.alias('__p3')])
    a = (pl.col('close') - sum_(pl.col('__p2'), 6)) / sum_(pl.col('__p1'), 6) * 288.0
    b = (pl.col('close') - sum_(pl.col('__p2'), 12)) / sum_(pl.col('__p3') - pl.col('__p2'), 12) * 144.0
    c = (pl.col('close') - sum_(pl.col('__p2'), 24)) / sum_(pl.col('__p1'), 24) * 144.0
    expr = ((a + b + c) * 100.0 / 504.0).alias('gtja_159')
    return df.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_159', impl=gtja_159, direction='normal', category='momentum', description='3-window cumulative range-position oscillator (KDJ-like)'))

def gtja_160(panel: pl.DataFrame) -> pl.Series:
    """GTJA #160 — SMA(C<=prev_C ? STD(C,20) : 0, 20, 1). Down-day vol EWMA."""
    pc = delay(pl.col('close'), 1)
    s = std_(pl.col('close'), 20)
    masked = pl.when(pc.is_null() | s.is_null()).then(None).when(pl.col('close') <= pc).then(s).otherwise(0.0)
    expr = sma(masked, 20, 1).alias('gtja_160')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_160', impl=gtja_160, direction='normal', category='volatility', description='EWMA of down-day std(close,20)'))

# ---- third_party/aurumq_gtja191/aurumq_rl/factors/gtja191/batch_161_180.py ----
"""GTJA-191 batch 161..180 (20 factors).

Special factors in this batch
-----------------------------

* ``gtja_165`` — Errata factor per ``wpwp/Alpha-101-GTJA-191`` README.
  We follow Daic115's pandas reference verbatim (which expands SUMAC to
  rolling sum of mean-deviations). Quality flag = 1.

* ``gtja_166`` — Errata factor (same source). Daic115 implements an
  approximation:
  ``5 * SUM(part-1-mean(part-1,20),20) / (SUM(mean(part,20)^2,20))^1.5``.
  We follow that. Quality flag = 1.
"""
import polars as pl

def gtja_161(panel: pl.DataFrame) -> pl.Series:
    """GTJA #161 — MEAN(MAX of 3 ATR components, 12). 12d average true range."""
    pc = delay(pl.col('close'), 1)
    a = pl.col('high') - pl.col('low')
    b = (pc - pl.col('high')).abs()
    c = (pc - pl.col('low')).abs()
    tr = pl.max_horizontal([pl.max_horizontal([a, b]), c])
    expr = mean(tr, 12).alias('gtja_161')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_161', impl=gtja_161, direction='normal', category='volatility', description='12-day mean true range'))

def gtja_162(panel: pl.DataFrame) -> pl.Series:
    """GTJA #162 — RSI-style normalised: (RSI - min(RSI,12)) / (max(RSI,12) - min(RSI,12))."""
    dc = pl.col('close') - delay(pl.col('close'), 1)
    p2 = sma(pl.when(dc > 0).then(dc).otherwise(0.0), 12, 1)
    p3 = sma(dc.abs(), 12, 1)
    rsi = p2 / p3 * 100.0
    df = panel.with_columns(rsi.alias('__rsi'))
    df = df.with_columns([ts_min(pl.col('__rsi'), 12).alias('__rmin'), ts_max(pl.col('__rsi'), 12).alias('__rmax')])
    expr = ((pl.col('__rsi') - pl.col('__rmin')) / (pl.col('__rmax') - pl.col('__rmin'))).alias('gtja_162')
    return df.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_162', impl=gtja_162, direction='normal', category='momentum', description='Stochastic-RSI: (RSI-min(RSI,12)) / (max(RSI,12)-min(RSI,12))'))

def gtja_163(panel: pl.DataFrame) -> pl.Series:
    """GTJA #163 — RANK(-RET * MEAN(V,20) * VWAP * (HIGH - CLOSE))."""
    ret = pl.col('close') / delay(pl.col('close'), 1) - 1.0
    inner = -1.0 * ret * mean(pl.col('volume'), 20) * pl.col('vwap') * (pl.col('high') - pl.col('close'))
    df = panel.with_columns(inner.alias('__i'))
    return df.select(rank(pl.col('__i')).alias('gtja_163')).to_series()
register_gtja191(FactorEntry(id='gtja_163', impl=gtja_163, direction='reverse', category='momentum', description='rank(-ret * mv20 * vwap * (high-close))'))

def gtja_164(panel: pl.DataFrame) -> pl.Series:
    """GTJA #164 — Daic115 reference (reciprocal-diff stochastic, SMA13 smoothed).

    Note: Daic115's parens in the original are slightly off — we follow
    their parsing literally for parity.
    """
    pc = delay(pl.col('close'), 1)
    diff = pl.col('close') - pc
    cond = pl.col('close') > pc
    rec = pl.when(cond).then(1.0 / diff).otherwise(1.0)
    inner = rec - ts_min(rec, 12) / (pl.col('high') - pl.col('low')) * 100.0
    expr = sma(inner, 13, 2).alias('gtja_164')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_164', impl=gtja_164, direction='normal', category='momentum', description='SMA((rec - min(rec,12) / (h-l) * 100), 13, 2) — Daic115 parsing'))

def gtja_165(panel: pl.DataFrame) -> pl.Series:
    """GTJA #165 — Errata (Daic115 expands SUMAC).

    Guotai Junan Formula (paper)
    ----------------------------
        MAX(SUMAC(CLOSE-MEAN(CLOSE,48))) - MIN(SUMAC(CLOSE-MEAN(CLOSE,48))) / STD(CLOSE,48)

    Daic115 expands SUMAC as ``SUM(diff, 48)`` and computes:
        TS_MAX(SUM(diff,48),48) - TS_MIN(SUM(diff,48),48) / STD(CLOSE,48)

    Listed in errata as ``return 0`` in some references. We follow
    Daic115's expansion. Quality flag = 1.

    Direction: ``normal``. Quality flag: ``1``.
    """
    diff = pl.col('close') - mean(pl.col('close'), 48)
    s48 = sum_(diff, 48)
    expr = (ts_max(s48, 48) - ts_min(s48, 48) / std_(pl.col('close'), 48)).alias('gtja_165')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_165', impl=gtja_165, direction='normal', category='volatility', description='MAX(SUMAC(close-ma48)) - MIN(SUMAC(close-ma48)) / STD(close,48) — Daic115 SUMAC expansion', quality_flag=1))

def gtja_166(panel: pl.DataFrame) -> pl.Series:
    """GTJA #166 — Errata (Daic115 simplification of skewness-of-returns).

    Guotai Junan Formula (paper, with errata)
    ------------------------------------------
        -20 * 19^1.5 * SUM(part1 - mean(part1,20), 20) /
        ((20-1)*(20-2)*(SUM((part^2,20))^1.5))
        where part = CLOSE / DELAY(CLOSE,1)

    Daic115 simplifies to:
        5 * SUM(part-1 - mean(part-1,20),20) / (SUM(mean(part,20)^2,20))^1.5

    Listed in errata. We follow Daic115's simplification. Quality flag = 1.

    Direction: ``normal``. Quality flag: ``1``.
    """
    part = pl.col('close') / delay(pl.col('close'), 1)
    p1 = part - 1.0
    df = panel.with_columns([(p1 - mean(p1, 20)).alias('__centered'), mean(part, 20).alias('__mp')])
    expr = (5.0 * sum_(pl.col('__centered'), 20) / sum_(pl.col('__mp').pow(2), 20).pow(1.5)).alias('gtja_166')
    return df.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_166', impl=gtja_166, direction='normal', category='momentum', description='5*sum(centered_ret,20)/(sum(ma_ret_squared,20))^1.5 — Daic115 errata simplification', quality_flag=1))

def gtja_167(panel: pl.DataFrame) -> pl.Series:
    """GTJA #167 — SUM(MAX(C-prev_C,0), 12). 12-day cumulative up-move."""
    dc = pl.col('close') - delay(pl.col('close'), 1)
    expr = sum_(pl.when(dc > 0).then(dc).otherwise(0.0), 12).alias('gtja_167')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_167', impl=gtja_167, direction='normal', category='momentum', description='12-day cumulative up-move'))

def gtja_168(panel: pl.DataFrame) -> pl.Series:
    """GTJA #168 — -V / MEAN(V,20). Inverse-volume-relative."""
    expr = (-1.0 * pl.col('volume') / mean(pl.col('volume'), 20)).alias('gtja_168')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_168', impl=gtja_168, direction='reverse', category='volume', description='-volume / mean(volume,20)'))

def gtja_169(panel: pl.DataFrame) -> pl.Series:
    """GTJA #169 — DEA-style on SMA-of-dC."""
    inner = sma(pl.col('close') - delay(pl.col('close'), 1), 9, 1)
    df = panel.with_columns(delay(inner, 1).alias('__d'))
    expr = sma(mean(pl.col('__d'), 12) - mean(pl.col('__d'), 26), 10, 1).alias('gtja_169')
    return df.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_169', impl=gtja_169, direction='normal', category='momentum', description='SMA(mean(delay(SMA(dC,9,1),1),12) - mean(delay(SMA(dC,9,1),1),26), 10, 1)'))

def gtja_170(panel: pl.DataFrame) -> pl.Series:
    """GTJA #170 — Weighted multi-rank composite."""
    df = panel.with_columns(rank(1.0 / pl.col('close')).alias('__r1'))
    df = df.with_columns(rank(pl.col('high') - pl.col('close')).alias('__r2'))
    a = pl.col('__r1') * pl.col('volume') / mean(pl.col('volume'), 20)
    b = pl.col('high') * pl.col('__r2') / (sum_(pl.col('high'), 5) / 5.0)
    df = df.with_columns((pl.col('vwap') - delay(pl.col('vwap'), 5)).alias('__dvwap5'))
    df = df.with_columns(rank(pl.col('__dvwap5')).alias('__r3'))
    expr = (a * b - pl.col('__r3')).alias('gtja_170')
    return df.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_170', impl=gtja_170, direction='normal', category='volume', description='rank(1/c)*v/mv20 * (h*rank(h-c))/(sum(h,5)/5) - rank(vwap-d_vwap_5)'))

def gtja_171(panel: pl.DataFrame) -> pl.Series:
    """GTJA #171 — -1 * (L-C) * O^5 / ((C-H)*C^5)."""
    expr = (-1.0 * (pl.col('low') - pl.col('close')) * pl.col('open').pow(5) / ((pl.col('close') - pl.col('high') + 1e-07) * pl.col('close').pow(5))).alias('gtja_171')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_171', impl=gtja_171, direction='reverse', category='momentum', description='-(low-close)*open^5 / ((close-high)*close^5)'))

def gtja_172(panel: pl.DataFrame) -> pl.Series:
    """GTJA #172 — DI-difference oscillator (DX) averaged over 6 days."""
    pc = delay(pl.col('close'), 1)
    a = pl.col('high') - pl.col('low')
    b = (pl.col('high') - pc).abs()
    c = (pl.col('low') - pc).abs()
    tr = pl.max_horizontal([pl.max_horizontal([a, b]), c])
    hd = pl.col('high') - delay(pl.col('high'), 1)
    ld = delay(pl.col('low'), 1) - pl.col('low')
    pos_ld = pl.when((ld > 0) & (ld > hd)).then(ld).otherwise(0.0)
    pos_hd = pl.when((hd > 0) & (hd > ld)).then(hd).otherwise(0.0)
    sum_tr = sum_(tr, 14)
    p1 = sum_(pos_ld, 14) * 100.0 / sum_tr
    p2 = sum_(pos_hd, 14) * 100.0 / sum_tr
    expr = mean((p1 - p2).abs() / (p1 + p2) * 100.0, 6).alias('gtja_172')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_172', impl=gtja_172, direction='normal', category='momentum', description='6-day mean of DI-difference oscillator (DX)'))

def gtja_173(panel: pl.DataFrame) -> pl.Series:
    """GTJA #173 — 3*SMA(C,13,2) - 2*SMA^2(C,13,2) + SMA^3(LOG(C),13,2)."""
    ma = sma(pl.col('close'), 13, 2)
    expr = (3.0 * ma - 2.0 * sma(ma, 13, 2) + sma(sma(log_(pl.col('close')), 13, 2), 13, 2)).alias('gtja_173')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_173', impl=gtja_173, direction='normal', category='momentum', description='3*sma(c,13,2) - 2*sma^2(c,13,2) + sma^3(log(c),13,2)'))

def gtja_174(panel: pl.DataFrame) -> pl.Series:
    """GTJA #174 — SMA(C>prev_C ? STD(C,20) : 0, 20, 1). Up-day vol EWMA."""
    pc = delay(pl.col('close'), 1)
    s = std_(pl.col('close'), 20)
    masked = pl.when(pc.is_null() | s.is_null()).then(None).when(pl.col('close') > pc).then(s).otherwise(0.0)
    expr = sma(masked, 20, 1).alias('gtja_174')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_174', impl=gtja_174, direction='normal', category='volatility', description='EWMA of up-day std(close,20)'))

def gtja_175(panel: pl.DataFrame) -> pl.Series:
    """GTJA #175 — 6-day mean true range."""
    pc = delay(pl.col('close'), 1)
    a = pl.col('high') - pl.col('low')
    b = (pc - pl.col('high')).abs()
    c = (pc - pl.col('low')).abs()
    tr = pl.max_horizontal([pl.max_horizontal([a, b]), c])
    expr = mean(tr, 6).alias('gtja_175')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_175', impl=gtja_175, direction='normal', category='volatility', description='6-day mean true range'))

def gtja_176(panel: pl.DataFrame) -> pl.Series:
    """GTJA #176 — CORR(RANK(stoch_K), RANK(VOLUME), 6)."""
    stoch = (pl.col('close') - ts_min(pl.col('low'), 12)) / (ts_max(pl.col('high'), 12) - ts_min(pl.col('low'), 12))
    df = panel.with_columns(stoch.alias('__stoch'))
    df = df.with_columns([rank(pl.col('__stoch')).alias('__rs'), rank(pl.col('volume')).alias('__rv')])
    expr = corr(pl.col('__rs'), pl.col('__rv'), 6).alias('gtja_176')
    return df.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_176', impl=gtja_176, direction='normal', category='correlation', description='corr(rank(stoch_K_12), rank(volume), 6)'))

def gtja_177(panel: pl.DataFrame) -> pl.Series:
    """GTJA #177 — (20 - HIGHDAY(HIGH,20)) / 20 * 100. Recency-of-high."""
    expr = ((20.0 - highday(pl.col('high'), 20)) / 20.0 * 100.0).alias('gtja_177')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_177', impl=gtja_177, direction='normal', category='momentum', description='(20-highday(high,20))/20*100 — recency-of-high oscillator'))

def gtja_178(panel: pl.DataFrame) -> pl.Series:
    """GTJA #178 — (C-prev_C)/prev_C * V. Vol-weighted daily return."""
    pc = delay(pl.col('close'), 1)
    expr = ((pl.col('close') - pc) / pc * pl.col('volume')).alias('gtja_178')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_178', impl=gtja_178, direction='normal', category='volume', description='(c-prev_c)/prev_c * volume'))

def gtja_179(panel: pl.DataFrame) -> pl.Series:
    """GTJA #179 — RANK(CORR(VWAP,V,4)) * RANK(CORR(RANK(LOW), RANK(MEAN(V,50)), 12))."""
    df = panel.with_columns([corr(pl.col('vwap'), pl.col('volume'), 4).alias('__c1'), mean(pl.col('volume'), 50).alias('__mv50'), rank(pl.col('low')).alias('__rl')])
    df = df.with_columns(rank(pl.col('__mv50')).alias('__rmv'))
    df = df.with_columns(corr(pl.col('__rl'), pl.col('__rmv'), 12).alias('__c2'))
    df = df.with_columns([rank(pl.col('__c1')).alias('__r1'), rank(pl.col('__c2')).alias('__r2')])
    return df.select((pl.col('__r1') * pl.col('__r2')).alias('gtja_179')).to_series()
register_gtja191(FactorEntry(id='gtja_179', impl=gtja_179, direction='normal', category='correlation', description='rank(corr(vwap,v,4)) * rank(corr(rank(low), rank(mv50), 12))'))

def gtja_180(panel: pl.DataFrame) -> pl.Series:
    """GTJA #180 — Conditional momentum vs negative volume.

    Guotai Junan Formula
    --------------------
        MEAN(VOLUME,20) < VOLUME ?
            -TSRANK(|DELTA(CLOSE,7)|,60) * SIGN(DELTA(CLOSE,7))
            : -VOLUME
    """
    df = panel.with_columns(delta(pl.col('close'), 7).alias('__d7'))
    cond = mean(pl.col('volume'), 20) < pl.col('volume')
    branch_true = -1.0 * ts_rank(pl.col('__d7').abs(), 60) * sign_(pl.col('__d7'))
    branch_false = -1.0 * pl.col('volume')
    expr = pl.when(cond).then(branch_true).otherwise(branch_false).alias('gtja_180')
    return df.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_180', impl=gtja_180, direction='reverse', category='momentum', description='Conditional: -tsrank(|d_c7|,60)*sign(d_c7) when V > MV20 else -V'))

# ---- third_party/aurumq_gtja191/aurumq_rl/factors/gtja191/batch_181_191.py ----
"""GTJA-191 batch 181..191 (11 factors).

Special factors in this batch
-----------------------------

* ``gtja_181`` — BENCHMARK + ERRATA. Daic115 reference uses cross-section
  mean of CLOSE returns as proxy for CSI300 (degraded data per
  alpha191.md handoff §4.1). We follow that for parity. Errata (paper
  formula has missing window argument). Quality flag = 1.

* ``gtja_182`` — BENCHMARK. Daic115 uses cross-section mean of OHLC as
  benchmark proxy. We follow that for parity. Quality flag = 0.

* ``gtja_183`` — Errata. SUMAC expansion (matches gtja_165 pattern).
  Quality flag = 1.

* ``gtja_191`` — Errata factor per ``wpwp/Alpha-101-GTJA-191`` README.
  Implement best-effort. Quality flag = 1.
"""
import polars as pl

def gtja_181(panel: pl.DataFrame) -> pl.Series:
    """GTJA #181 — Skewness-adjusted return-vs-benchmark composite (errata).

    Guotai Junan Formula
    --------------------
        SUM(((CLOSE/DELAY(CLOSE,1)-1) - MEAN(C/Cprev-1, 20)) -
            (BMK - MEAN(BMK,20))^2, 20) /
        SUM((BMK - MEAN(BMK,20))^3)

    Benchmark sourcing
    ------------------
    Same proxy approach as :func:`gtja_149` — Daic115's reference uses
    cross-section mean of CLOSE returns. Production wiring to CSI300
    OHLC is Phase D.

    Direction: ``normal``. Quality flag: ``1`` (errata + degraded data).
    """
    ret = pl.col('close') / delay(pl.col('close'), 1) - 1.0
    df = panel.with_columns(ret.alias('__ret'))
    bench = pl.col('__ret').mean().over('trade_date')
    df = df.with_columns(bench.alias('__bench'))
    df = df.with_columns([(pl.col('__ret') - mean(pl.col('__ret'), 20)).alias('__centered'), (pl.col('__bench') - mean(pl.col('__bench'), 20)).alias('__bcent')])
    num = sum_(pl.col('__centered') - pl.col('__bcent').pow(2), 20)
    den = sum_(pl.col('__bcent').pow(3), 20)
    expr = (num / den).alias('gtja_181')
    return df.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_181', impl=gtja_181, direction='normal', category='benchmark', description='Skew-adjusted return-vs-benchmark — CSI300 proxied via CS mean (Phase D)', quality_flag=1))

def gtja_182(panel: pl.DataFrame) -> pl.Series:
    """GTJA #182 — Co-movement count: (C>O & BMK_C>BMK_O) | (C<O & BMK_C<BMK_O).

    Guotai Junan Formula
    --------------------
        COUNT(
            (CLOSE > OPEN & BMK_C > BMK_O) | (CLOSE < OPEN & BMK_C < BMK_O),
            20
        ) / 20

    Benchmark sourcing — cross-section mean of OHLC (matches Daic115).
    Phase D: switch to true CSI300.

    Direction: ``normal``. Quality flag: ``0``.
    """
    df = panel.with_columns([pl.col('close').mean().over('trade_date').alias('__bc'), pl.col('open').mean().over('trade_date').alias('__bo')])
    bench_up = pl.col('__bc') > pl.col('__bo')
    stock_up = pl.col('close') > pl.col('open')
    same_dir = stock_up == bench_up
    expr = (sum_(same_dir.cast(pl.Float64), 20) / 20.0).alias('gtja_182')
    return df.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_182', impl=gtja_182, direction='normal', category='benchmark', description='20d frac of days stock & benchmark co-move — CS-mean OHLC proxy (Phase D)', quality_flag=0))

def gtja_183(panel: pl.DataFrame) -> pl.Series:
    """GTJA #183 — Errata (Daic115 SUMAC expansion).

    Guotai Junan Formula
    --------------------
        MAX(SUMAC(C-MEAN(C,24))) - MIN(SUMAC(C-MEAN(C,24))) / STD(C,24)

    Daic115 expands SUMAC = SUM(diff, 24). Quality flag = 1.

    Direction: ``normal``. Quality flag: ``1``.
    """
    diff = pl.col('close') - mean(pl.col('close'), 24)
    s = sum_(diff, 24)
    expr = (ts_max(s, 24) - ts_min(s, 24) / std_(pl.col('close'), 24)).alias('gtja_183')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_183', impl=gtja_183, direction='normal', category='volatility', description='MAX(SUMAC(c-ma24,24)) - MIN(SUMAC(c-ma24,24)) / STD(c,24)', quality_flag=1))

def gtja_184(panel: pl.DataFrame) -> pl.Series:
    """GTJA #184 — RANK(CORR(DELAY(O-C,1), C, 200)) + RANK(O-C)."""
    df = panel.with_columns([delay(pl.col('open') - pl.col('close'), 1).alias('__d'), (pl.col('open') - pl.col('close')).alias('__oc')])
    df = df.with_columns(corr(pl.col('__d'), pl.col('close'), 200).alias('__c'))
    df = df.with_columns([rank(pl.col('__c')).alias('__r1'), rank(pl.col('__oc')).alias('__r2')])
    return df.select((pl.col('__r1') + pl.col('__r2')).alias('gtja_184')).to_series()
register_gtja191(FactorEntry(id='gtja_184', impl=gtja_184, direction='normal', category='correlation', description='rank(corr(delay(o-c,1), close, 200)) + rank(o-c)'))

def gtja_185(panel: pl.DataFrame) -> pl.Series:
    """GTJA #185 — RANK(-1 * (1 - O/C)^2). Squared open-close gap, ranked."""
    inner = -1.0 * (1.0 - pl.col('open') / pl.col('close')).pow(2)
    df = panel.with_columns(inner.alias('__i'))
    return df.select(rank(pl.col('__i')).alias('gtja_185')).to_series()
register_gtja191(FactorEntry(id='gtja_185', impl=gtja_185, direction='reverse', category='momentum', description='rank(-(1 - open/close)^2)'))

def gtja_186(panel: pl.DataFrame) -> pl.Series:
    """GTJA #186 — Smoothed DI-difference oscillator."""
    pc = delay(pl.col('close'), 1)
    a = pl.col('high') - pl.col('low')
    b = (pl.col('high') - pc).abs()
    c = (pl.col('low') - pc).abs()
    tr = pl.max_horizontal([pl.max_horizontal([a, b]), c])
    hd = pl.col('high') - delay(pl.col('high'), 1)
    ld = delay(pl.col('low'), 1) - pl.col('low')
    pos_ld = pl.when((ld > 0) & (ld > hd)).then(ld).otherwise(0.0)
    pos_hd = pl.when((hd > 0) & (hd > ld)).then(hd).otherwise(0.0)
    sum_tr = sum_(tr, 14)
    p1 = sum_(pos_ld, 14) * 100.0 / sum_tr
    p2 = sum_(pos_hd, 14) * 100.0 / sum_tr
    p3 = (p1 - p2).abs() / (p1 + p2) * 100.0
    expr = ((mean(p3, 6) + delay(mean(p3, 6), 6)) / 2.0).alias('gtja_186')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_186', impl=gtja_186, direction='normal', category='momentum', description='(mean(DX,6) + delay(mean(DX,6),6)) / 2 — smoothed ADX-style'))

def gtja_187(panel: pl.DataFrame) -> pl.Series:
    """GTJA #187 — SUM(O<=prev_O ? 0 : MAX(H-O, O-prev_O), 20). Open-gap up cumsum."""
    po = delay(pl.col('open'), 1)
    body = pl.max_horizontal([pl.col('high') - pl.col('open'), pl.col('open') - po])
    masked = pl.when(pl.col('open') <= po).then(0.0).otherwise(body)
    expr = sum_(masked, 20).alias('gtja_187')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_187', impl=gtja_187, direction='normal', category='momentum', description='20-day cumulative open-up-gap or daily upper-body-range'))

def gtja_188(panel: pl.DataFrame) -> pl.Series:
    """GTJA #188 — ((H-L) - SMA(H-L,11,2)) / SMA(H-L,11,2) * 100. Range deviation."""
    rng = pl.col('high') - pl.col('low')
    s = sma(rng, 11, 2)
    expr = ((rng - s) / s * 100.0).alias('gtja_188')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_188', impl=gtja_188, direction='normal', category='volatility', description='((H-L) - SMA(H-L,11,2)) / SMA(H-L,11,2) * 100'))

def gtja_189(panel: pl.DataFrame) -> pl.Series:
    """GTJA #189 — MEAN(|C - MEAN(C,6)|, 6). Mean abs deviation from MA6."""
    expr = mean((pl.col('close') - mean(pl.col('close'), 6)).abs(), 6).alias('gtja_189')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_189', impl=gtja_189, direction='normal', category='volatility', description='6-day mean abs deviation from MA6'))

def gtja_190(panel: pl.DataFrame) -> pl.Series:
    """GTJA #190 — Log-asymmetric return classifier.

    Guotai Junan Formula
    --------------------
        LOG((COUNT(p1>p2,20)-1) * SUMIF((p1-p2)^2,20,p1<p2) /
            ((COUNT(p1<p2,20)) * SUMIF((p1-p2)^2,20,p1>p2)))
        where p1 = C/prev_C - 1, p2 = (C/C-19)^(1/20) - 1
    """
    pc = delay(pl.col('close'), 1)
    pc19 = delay(pl.col('close'), 19)
    p1 = pl.col('close') / pc - 1.0
    p2 = (pl.col('close') / pc19).pow(1.0 / 20.0) - 1.0
    df = panel.with_columns([p1.alias('__p1'), p2.alias('__p2')])
    df = df.with_columns((pl.col('__p1') - pl.col('__p2')).pow(2).alias('__sq'))
    cond_gt = pl.col('__p1') > pl.col('__p2')
    cond_lt = pl.col('__p1') < pl.col('__p2')
    sumif_lt = sumif(pl.col('__sq'), 20, cond_lt)
    sumif_gt = sumif(pl.col('__sq'), 20, cond_gt)
    cnt_gt = count_(cond_gt, 20)
    cnt_lt = count_(cond_lt, 20)
    expr = log_((cnt_gt - 1.0) * sumif_lt / (cnt_lt * sumif_gt + 1e-12)).alias('gtja_190')
    return df.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_190', impl=gtja_190, direction='normal', category='momentum', description='log((cnt_p1>p2-1)*SUMIF((p1-p2)^2,20,p1<p2)/(cnt_p1<p2 * SUMIF(.,20,p1>p2)))'))

def gtja_191(panel: pl.DataFrame) -> pl.Series:
    """GTJA #191 — Errata (best-effort).

    Guotai Junan Formula
    --------------------
        CORR(MEAN(VOLUME,20), LOW, 5) + (HIGH+LOW)/2 - CLOSE

    Listed in errata. The formula is well-defined; we implement it as
    Daic115 does. Quality flag = 1.

    Direction: ``normal``. Quality flag: ``1``.
    """
    expr = (corr(mean(pl.col('volume'), 20), pl.col('low'), 5) + (pl.col('high') + pl.col('low')) / 2.0 - pl.col('close')).alias('gtja_191')
    return panel.select(expr).to_series()
register_gtja191(FactorEntry(id='gtja_191', impl=gtja_191, direction='normal', category='volume', description='corr(mv20, low, 5) + (h+l)/2 - close', quality_flag=1))

REGISTRY = GTJA191_REGISTRY
