import random
from transformers import CLIPTokenizer

START_OF_TEXT_ID = 49406
END_OF_TEXT_ID = 49407
COMMA_ID = 267
BRACKET_COMMA_ID = 2361
DELIMITER_IDS = {COMMA_ID, BRACKET_COMMA_ID}
MAX_LENGTH = 227


def parse_input_ids(
    token_ids: list[int], exclude_special_tokens: bool = True
) -> list[list[int]]:
    """
    Parses tokenized prompts by segmenting them into lists of tags using
    delimiter IDs (commas or brackets + commas).
    """
    core_tokens = token_ids
    if exclude_special_tokens:
        core_tokens = core_tokens[1:-1]

    tags = []
    current_tag = []
    for token in core_tokens:
        if token == COMMA_ID:
            if current_tag:
                tags.append(current_tag)
            current_tag = []
        elif token == BRACKET_COMMA_ID:
            if current_tag:
                current_tag.append(token)
                tags.append(current_tag)
            current_tag = []
        else:
            current_tag.append(token)

    if current_tag:
        tags.append(current_tag)

    return tags


def reconstruct_input_ids(
    tags: list[list[int]],
    include_special_tokens: bool = True,
    include_comma_last: bool = False,
) -> list[int]:
    """
    Reassembles individual token segments back into a unified flat sequence
    retaining standard prompt commas and special tokens.
    """
    shuffled_core_tokens = []
    num_tags = len(tags)
    last_tag = 0 if include_comma_last else 1
    for i, tag in enumerate(tags):
        shuffled_core_tokens.extend(tag)
        if i < num_tags - last_tag and tag[-1] != BRACKET_COMMA_ID:
            shuffled_core_tokens.append(COMMA_ID)
    if not include_special_tokens:
        return shuffled_core_tokens
    return [START_OF_TEXT_ID] + shuffled_core_tokens + [END_OF_TEXT_ID]


def shuffle_prompt_token_ids(
    token_ids: list[int],
    drop_prob: float = 0.0,
    prompt_len: int = 0,
    include_special_tokens: bool = True,
) -> list[int]:
    """
    Randomizes order of token segments (tags) in a prompt while maintaining
    delimiters and performing optional probabilistic tag dropout.
    """
    tags = parse_input_ids(token_ids, include_special_tokens)
    tags_len = [len(tag) for tag in tags]

    if drop_prob > 0.0:
        tags_to_keep = []
        tags_len_to_keep = []
        for t, t_len in zip(tags, tags_len):
            if random.random() >= drop_prob:
                tags_to_keep.append(t)
                tags_len_to_keep.append(t_len)
            else:
                prompt_len -= t_len
        tags = tags_to_keep
        tags_len = tags_len_to_keep

    tokens_to_free = prompt_len - MAX_LENGTH
    if tokens_to_free > 0:
        removed_tokens = 0
        while tokens_to_free > 0 and tags:
            tag_to_remove_idx = random.randrange(len(tags))
            tags.pop(tag_to_remove_idx)
            removed_tokens += tags_len[tag_to_remove_idx]
            tokens_to_free -= removed_tokens

    random.shuffle(tags)

    shuffled_tokens = reconstruct_input_ids(
        tags,
        include_special_tokens,
        include_comma_last=not include_special_tokens,
    )
    return shuffled_tokens


def split_upsampled_tags(
    upsampled_tags: str,
    free_tokens: int,
    tokenizer,
    return_last_token: bool = False,
) -> str | tuple[str, int | None]:
    """
    Slices upsampled text tags to fit remaining token allocation size in
    sequence length constraints.
    """
    if free_tokens < 1:
        if return_last_token:
            return "", None
        return ""

    upsampled_tokens = tokenizer(
        upsampled_tags,
        padding=False,
        truncation=False,
        return_length=True,
    )
    if (upsampled_tokens["length"] - 2) < free_tokens:
        if return_last_token:
            last_token = upsampled_tokens["input_ids"][-2]
            return upsampled_tags, last_token
        else:
            return upsampled_tags

    tags = parse_input_ids(upsampled_tokens["input_ids"])
    included_tags = []
    remaining_tokens = free_tokens
    for tag in tags:
        tag_len = len(tag)
        if tag_len < remaining_tokens:
            included_tags.append(tag)
            remaining_tokens -= tag_len
        else:
            break

    sliced_input_ids = reconstruct_input_ids(included_tags)
    decoded_sliced_input_ids = tokenizer.decode(
        sliced_input_ids, skip_special_tokens=True
    )
    if return_last_token:
        last_token = sliced_input_ids[-2] if sliced_input_ids else None
        return decoded_sliced_input_ids, last_token
    return decoded_sliced_input_ids
