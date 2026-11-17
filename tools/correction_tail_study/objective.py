"""Use the same frozen radar-tail surrogate on the original CorrectionNet."""
from tools.unet_ablation.models import objective as baseline_objective
from tools.unet_tail_loss.objective import matched_filter_tail_loss, TAIL_WEIGHT
from .settings import ARM


def study_objective(correction, batch, records):
    base = baseline_objective(correction, batch['s'], batch['j'], batch['e'], 'full')
    # Compute the same diagnostic for both arms; only its training weight differs.
    tail, diagnostics = matched_filter_tail_loss(
        correction.reshape(records, 8, 4, 2, 1800), batch['j'], batch['e'])
    weight = TAIL_WEIGHT if ARM == 'tail' else 0.0
    return {**base, **diagnostics, 'tail': tail,
            'total': base['total'] + weight * tail}
