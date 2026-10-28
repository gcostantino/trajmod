"""Transfer of nonlinear parameters (postseismic taus, SSE times) between components."""
import datetime
import json
from typing import Dict, List, Optional

import numpy as np

PRIOR_VERSION = 1


def day_key(x) -> str:
    """Calendar day of a datetime-like value as 'YYYY-MM-DD' (str() for anything else)."""
    if isinstance(x, (datetime.datetime, datetime.date)):
        return x.strftime('%Y-%m-%d')
    if isinstance(x, np.datetime64):
        return str(np.datetime64(x, 'D'))
    return str(x)


def _iso(x) -> str:
    if isinstance(x, datetime.datetime):
        return x.isoformat()
    return str(np.datetime64(x, 's')) if isinstance(x, np.datetime64) else str(x)


def build_prior(post_specs: List[Dict], selected_sse: List[Dict],
                catalog_sse: List[Dict]) -> Dict:
    """Nonlinear parameters of a fitted model, JSON-serialisable.

    selected_sse and catalog_sse must be in the same order (fitted vs catalogue times).
    """
    post: Dict[str, List[Dict]] = {}
    for s in post_specs:
        post.setdefault(day_key(s['eq_day']), []).append(
            {'kind': s.get('kind', 'log'), 'tau': float(s['tau'])})
    sse = [{'catalog_start': day_key(c['start_day']), 'catalog_end': day_key(c['end_day']),
            'start': _iso(f['start_day']), 'end': _iso(f['end_day'])}
           for c, f in zip(catalog_sse, selected_sse)]
    return {'version': PRIOR_VERSION,
            'postseismic': [{'eq_day': k, 'terms': v} for k, v in post.items()],
            'sse': sse}


def validate_prior(prior: Dict) -> None:
    if not isinstance(prior, dict) or prior.get('version') != PRIOR_VERSION:
        raise ValueError(f"nonlinear_prior must be a dict from export_nonlinear() "
                         f"(version {PRIOR_VERSION})")
    for key in ('postseismic', 'sse'):
        if key not in prior:
            raise ValueError(f"nonlinear_prior is missing '{key}'")


def prior_terms_for(prior: Optional[Dict], eq_day) -> Optional[List[Dict]]:
    """[{'kind', 'tau'}, ...] of the master for this earthquake day, or None."""
    if prior is None:
        return None
    k = day_key(eq_day)
    for p in prior['postseismic']:
        if p['eq_day'] == k:
            return p['terms']
    return None


def prior_sse_for(prior: Optional[Dict], start_day, end_day) -> Optional[Dict]:
    """Master entry whose catalogue start and end match this SSE, or None."""
    if prior is None:
        return None
    ks, ke = day_key(start_day), day_key(end_day)
    for p in prior['sse']:
        if p['catalog_start'] == ks and p['catalog_end'] == ke:
            return p
    return None


def save_prior(prior: Dict, path: str) -> None:
    with open(path, 'w') as f:
        json.dump(prior, f, indent=2)


def load_prior(path: str) -> Dict:
    with open(path) as f:
        prior = json.load(f)
    validate_prior(prior)
    return prior
