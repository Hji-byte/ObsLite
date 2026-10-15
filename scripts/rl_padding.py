"""Drop only all-masked prompt prefixes before dense actor forward passes."""


def trim_masked_prompt(input_ids, attention_mask, position_ids, response_length):
    prompt_length = input_ids.shape[-1] - response_length
    if prompt_length <= 0:
        raise ValueError('Expected a nonempty prompt before the response')
    active_columns = attention_mask[:, :prompt_length].bool().any(dim=0).nonzero()
    if active_columns.numel() == 0:
        raise ValueError('Cannot train an entirely masked prompt')
    start = int(active_columns[0, 0])
    return input_ids[:, start:], attention_mask[:, start:], position_ids[..., start:]
