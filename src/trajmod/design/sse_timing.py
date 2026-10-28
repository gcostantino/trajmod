"""Bounded optimisation of SSE start/end times (daily GNSS, whole-day shifts).

The linear coefficients are profiled out exactly: for a candidate (start, end)
of one SSE, the weighted chi-square equals ||r_y||^2 - (r_c . r_y)^2 / ||r_c||^2,
where r_y and r_c are the data and the candidate column projected onto the
orthogonal complement of all the other columns. No refit is needed per candidate.
"""
import datetime
import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SSETimingResult:
    """Outcome of the SSE timing optimisation."""
    events: List[Dict]  # SSE dicts (same order as input), shifted where accepted
    report: List[Dict]  # one entry per SSE
    chi2_catalog: float  # weighted chi2 with the catalog times
    chi2_final: float  # weighted chi2 with the final times
    n_obs: int
    n_moved: int  # number of SSEs whose timing changed
    n_passes: int
    delta_ic: float  # IC(final) - IC(catalog); negative = improvement
    criterion: str = 'bic'


def sse_tolerance_days(start_day: datetime.datetime,
                       end_day: datetime.datetime,
                       config) -> float:
    """Per-event tolerance: min(max_days, max(tol_days, frac * duration))."""
    duration = max(0.0, (end_day - start_day).total_seconds() / 86400.0)
    tol = max(config.sse_timing_tol_days, config.sse_timing_tol_frac * duration)
    return float(min(config.sse_timing_tol_max_days, tol))


def _raised_cosine_matrix(tm: np.ndarray, s: float, ends: np.ndarray) -> np.ndarray:
    """Raised-cosine columns for one start `s` and several ends (day offsets).

    Same function as TemplateFunctions.raised_cosine_days, vectorised over ends.
    Columns with end <= start are all zeros (as in the template).
    """
    t = tm[:, None]
    E = np.atleast_1d(np.asarray(ends, dtype=float))[None, :]
    dur = E - s
    ok = dur > 0
    safe = np.where(ok, dur, 1.0)
    phase = np.clip((t - s) / safe, 0.0, 1.0)
    out = 0.5 * (1.0 - np.cos(np.pi * phase))
    return np.where(ok, out, 0.0)


def _orth_basis(A: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the column space of A (rank-revealing via SVD)."""
    if A.shape[1] == 0:
        return np.zeros((A.shape[0], 0))
    U, S, _ = np.linalg.svd(A, full_matrices=False)
    if S.size == 0 or S[0] == 0:
        return np.zeros((A.shape[0], 0))
    return U[:, S > 1e-10 * S[0]]


def _full_chi2(Ao: np.ndarray, yw: np.ndarray, tm: np.ndarray, wm: np.ndarray,
               starts: np.ndarray, ends: np.ndarray) -> float:
    """Weighted chi2 of the full linear model for given SSE times."""
    cols = [Ao] if Ao.shape[1] else []
    for s, e in zip(starts, ends):
        cols.append((_raised_cosine_matrix(tm, s, np.array([e]))[:, 0] * wm)[:, None])
    A = np.hstack(cols) if cols else np.zeros((len(yw), 0))
    if A.shape[1] == 0:
        return float(np.sum(yw ** 2))
    coeffs, *_ = np.linalg.lstsq(A, yw, rcond=None)
    r = yw - A @ coeffs
    return float(np.sum(r ** 2))


def _shift_event(ev: Dict, ds: int, de: int) -> Dict:
    """Copy of an SSE dict with start/end shifted by whole days."""
    new = dict(ev)
    new['orig_start_day'] = ev['start_day']
    new['orig_end_day'] = ev['end_day']
    new['start_day'] = ev['start_day'] + datetime.timedelta(days=int(ds))
    new['end_day'] = ev['end_day'] + datetime.timedelta(days=int(de))
    for key, shift in (('start', ds), ('end', de)):
        if key in new and isinstance(new[key], datetime.datetime):
            new[key] = new[key] + datetime.timedelta(days=int(shift))
    if 'duration' in new:
        new['duration'] = (new['end_day'] - new['start_day']).days
    new['timing_optimized'] = bool(ds != 0 or de != 0)
    return new


def optimize_sse_timing(t_days: np.ndarray,
                        t0: datetime.datetime,
                        y: np.ndarray,
                        weights: np.ndarray,
                        G_other: np.ndarray,
                        sse_events: List[Dict],
                        config) -> SSETimingResult:
    """Optimise SSE start/end times with all other columns held fixed.

    Args:
        t_days: observation times in days since t0 (one row per day)
        t0: reference datetime
        y: observations (NaN allowed)
        weights: w = 1/sigma (non-finite -> 0)
        G_other: all NON-SSE columns (baseline, EQ steps, non-EQ steps, postseismic), shape (N, p)
        sse_events: SSE dicts with 'start_day' and 'end_day' datetimes
        config: ModelConfig (sse_timing_* fields)

    Returns:
        SSETimingResult
    """
    n_sse = len(sse_events)
    t_days = np.asarray(t_days, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.array(weights, dtype=float)
    w[~np.isfinite(w)] = 0.0

    mask = np.isfinite(y) & (w > 0)
    tm, wm = t_days[mask], w[mask]
    yw = y[mask] * wm
    n_obs = int(mask.sum())
    Ao = (G_other[mask] * wm[:, None]) if G_other.shape[1] else np.zeros((n_obs, 0))

    pen = np.log(max(n_obs, 2)) if config.sse_timing_criterion == 'bic' else 2.0

    if n_sse == 0 or n_obs == 0:
        return SSETimingResult(events=[dict(e) for e in sse_events], report=[],
                               chi2_catalog=float(np.sum(yw ** 2)),
                               chi2_final=float(np.sum(yw ** 2)),
                               n_obs=n_obs, n_moved=0, n_passes=0, delta_ic=0.0,
                               criterion=config.sse_timing_criterion)

    step = float(config.sse_timing_step_days)
    min_dur = float(config.sse_min_duration_days)

    s0 = np.array([(ev['start_day'] - t0).days for ev in sse_events], dtype=float)
    e0 = np.array([(ev['end_day'] - t0).days for ev in sse_events], dtype=float)
    tol = np.array([sse_tolerance_days(ev['start_day'], ev['end_day'], config)
                    for ev in sse_events])
    K = np.floor(tol / step).astype(int)
    sigma = np.maximum(tol / 2.0, 1.0)  # prior width in days

    cur_s, cur_e = s0.copy(), e0.copy()
    Sw = np.column_stack([
        _raised_cosine_matrix(tm, cur_s[k], np.array([cur_e[k]]))[:, 0] * wm
        for k in range(n_sse)])

    chi2_catalog = _full_chi2(Ao, yw, tm, wm, s0, e0)

    n_passes = 0
    for p in range(config.sse_timing_max_rounds):
        n_passes = p + 1
        moved = False

        for k in range(n_sse):
            if K[k] == 0:
                continue

            others = [j for j in range(n_sse) if j != k]
            A = np.hstack([Ao, Sw[:, others]]) if others else Ao
            Q = _orth_basis(A)
            r_y = yw - Q @ (Q.T @ yw)
            ss_y = float(r_y @ r_y)

            # neighbours by current start time
            order = np.argsort(cur_s)
            i = int(np.where(order == k)[0][0])
            prev_end = cur_e[order[i - 1]] if i > 0 else -np.inf
            next_start = cur_s[order[i + 1]] if i < n_sse - 1 else np.inf

            grid = step * np.arange(-K[k], K[k] + 1)
            S_all = s0[k] + grid
            E_all = e0[k] + grid
            #lo_s = max(s0[k] - K[k] * step, min(prev_end, cur_s[k]))
            #hi_e = min(e0[k] + K[k] * step, max(next_start, cur_e[k]))
            if config.sse_timing_allow_overlap:
                lo_s = s0[k] - K[k] * step
                hi_e = e0[k] + K[k] * step
            else:
                lo_s = max(s0[k] - K[k] * step, min(prev_end, cur_s[k]))
                hi_e = min(e0[k] + K[k] * step, max(next_start, cur_e[k]))

            S_c = S_all[S_all >= lo_s]
            E_c = E_all[E_all <= hi_e]

            best_J, best_se, best_col = np.inf, None, None
            J_cur = np.inf

            for s in S_c:
                keep = ((E_c - s) >= min_dur) | ((s == cur_s[k]) & (E_c == cur_e[k]))
                E_s = E_c[keep]
                if E_s.size == 0:
                    continue

                C = _raised_cosine_matrix(tm, s, E_s) * wm[:, None]
                RC = C - Q @ (Q.T @ C) if Q.shape[1] else C
                den = np.einsum('ij,ij->j', RC, RC)
                num = (RC.T @ r_y) ** 2
                scale = np.einsum('ij,ij->j', C, C)
                good = den > 1e-8 * np.maximum(scale, 1e-300)
                chi2 = np.full(E_s.size, np.inf)
                chi2[good] = ss_y - num[good] / den[good]

                J = chi2.copy()
                if config.sse_timing_prior:
                    J = J + ((s - s0[k]) / sigma[k]) ** 2 + ((E_s - e0[k]) / sigma[k]) ** 2

                j = int(np.argmin(J))
                if J[j] < best_J:
                    best_J, best_se, best_col = J[j], (s, E_s[j]), C[:, j].copy()

                is_cur = (s == cur_s[k]) & (E_s == cur_e[k])
                if is_cur.any():
                    J_cur = float(J[np.argmax(is_cur)])

            if best_se is None or not np.isfinite(best_J):
                continue
            if (best_se[0] == cur_s[k] and best_se[1] == cur_e[k]):
                continue
            if best_J < J_cur - 1e-9 * max(1.0, abs(J_cur)) or not np.isfinite(J_cur):
                logger.debug(f"SSE {k}: ({cur_s[k]:.0f}, {cur_e[k]:.0f}) -> "
                             f"({best_se[0]:.0f}, {best_se[1]:.0f})")
                cur_s[k], cur_e[k] = best_se
                Sw[:, k] = best_col
                moved = True

        if not moved:
            break

    # ---- accept each moved SSE only if it beats the information-criterion penalty ----
    chi2_cur = _full_chi2(Ao, yw, tm, wm, cur_s, cur_e)
    accepted = np.ones(n_sse, dtype=bool)

    for k in range(n_sse):
        dk = int(cur_s[k] != s0[k]) + int(cur_e[k] != e0[k])
        if dk == 0:
            continue
        rs, re_ = cur_s.copy(), cur_e.copy()
        rs[k], re_[k] = s0[k], e0[k]
        chi2_rev = _full_chi2(Ao, yw, tm, wm, rs, re_)
        gain = n_obs * np.log(max(chi2_rev, 1e-300) / max(chi2_cur, 1e-300))
        if gain <= pen * dk:
            logger.info(f"SSE {k}: timing change not supported "
                        f"(gain={gain:.2f} <= {pen * dk:.2f}); reverted to catalog")
            cur_s, cur_e = rs, re_
            chi2_cur = chi2_rev
            accepted[k] = False

    chi2_final = chi2_cur

    events, report = [], []
    n_moved = 0
    total_dk = 0
    for k, ev in enumerate(sse_events):
        ds, de = int(cur_s[k] - s0[k]), int(cur_e[k] - e0[k])
        moved_k = (ds != 0) or (de != 0)
        n_moved += int(moved_k)
        total_dk += int(ds != 0) + int(de != 0)
        events.append(_shift_event(ev, ds, de))
        report.append({
            'sse_idx': k,
            'start_catalog': ev['start_day'],
            'end_catalog': ev['end_day'],
            'start_final': events[-1]['start_day'],
            'end_final': events[-1]['end_day'],
            'shift_start_days': ds,
            'shift_end_days': de,
            'tolerance_days': float(tol[k]),
            'at_bound': bool(abs(ds) >= K[k] * step > 0 or abs(de) >= K[k] * step > 0),
            'moved': moved_k,
        })

    delta_ic = (n_obs * np.log(max(chi2_final, 1e-300) / max(chi2_catalog, 1e-300))
                + pen * total_dk)

    logger.info(f"SSE timing: {n_moved}/{n_sse} moved, chi2 {chi2_catalog:.1f} -> "
                f"{chi2_final:.1f}, d{config.sse_timing_criterion.upper()}={delta_ic:.1f}")

    return SSETimingResult(events=events, report=report,
                           chi2_catalog=chi2_catalog, chi2_final=chi2_final,
                           n_obs=n_obs, n_moved=n_moved, n_passes=n_passes,
                           delta_ic=float(delta_ic),
                           criterion=config.sse_timing_criterion)
