"""Train new frames after a detached prediction-only overlapping window."""
import torch

from model.rnn.online import NON_TEMPORAL_KEYS, slice_time_value


def forward_with_ot_history(module, hands, contact, obj_init, window, **kwargs):
    batch_size, seq_len = hands.shape[:2]
    if not window < seq_len < 2 * window:
        raise ValueError("history training needs window < sequence length < 2 * window")

    def sliced(start, stop):
        return {key: value if key in NON_TEMPORAL_KEYS else slice_time_value(value, start, stop, batch_size, seq_len)
                for key, value in kwargs.items()}

    with torch.no_grad():
        warmup = module(hands[:, :window], contact[:, :window], obj_init, **sliced(0, window))
    start = seq_len - window
    history_end = window - start
    known_position = warmup['pred_obj_trans'][:, start:window].detach()
    known_weights = warmup['gating_weights'][:, start:window].detach()
    base_module = module.module if isinstance(module, torch.nn.DataParallel) else module
    state_keys = getattr(base_module, 'SEGMENT_STATE_OUTPUT_KEYS', ())
    state_kwargs = {}
    if state_keys:
        state_kwargs['known_segment_state_prefix'] = {key: warmup[key][:, start:window].detach() for key in state_keys}
    next_kwargs = sliced(start, seq_len)
    next_kwargs['contact_init'] = contact[:, start].detach()
    result = module(hands[:, start:], contact[:, start:], known_position[:, 0],
                    known_obj_trans_prefix=known_position, known_gating_weights_prefix=known_weights,
                    **state_kwargs, **next_kwargs)
    merged = {}
    for key, value in result.items():
        if key not in NON_TEMPORAL_KEYS and isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[:2] == (batch_size, window):
            merged[key] = torch.cat((warmup[key][:, :start].detach(), value), dim=1)
        else:
            merged[key] = value
    loss_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=hands.device)
    loss_mask[:, start + history_end:] = True
    merged['ot_loss_mask'] = loss_mask
    return merged