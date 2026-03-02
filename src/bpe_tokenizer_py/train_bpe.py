from collections import defaultdict
from operator import itemgetter

import regex as re

from bpe_tokenizer_py.pretokenization import find_chunk_boundaries, NUM_CHUNKS


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = "<|endoftext|>".encode("utf-8")
    pretoken_counts = defaultdict(int)
    merges = []
    with open(input_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(f, NUM_CHUNKS, b"<|endoftext|>")
        for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            split_chunks = re.split("|".join(map(re.escape, special_tokens)), chunk)
            # TODO:  Parallelize on chunk boundaries with multiprocessing
            for split_chunk in split_chunks:
                pretokens = re.finditer(PAT, split_chunk)
                for pretoken in pretokens:
                    pretoken_counts[
                        tuple(bytes([c]) for c in pretoken[0].encode("utf-8"))
                    ] += 1
    # TODO:  Pre-create candidate_pairs dictionary
    while len(vocab) < vocab_size:
        candidate_pairs = defaultdict(int)
        for pretoken, pretoken_count in pretoken_counts.items():
            # TODO:  Increment/decrement candidate_pairs at token boundaries
            for i in range(len(pretoken) - 1):
                candidate_pairs[(pretoken[i], pretoken[i + 1])] += pretoken_count
        # Get max first on pair count in corpus, then on pair itself
        # to get lexicographically greater pair
        max_pair_key = max(candidate_pairs.items(), key=itemgetter(1, 0))[0]
        merges.append(max_pair_key)
        new_byte = b"".join(max_pair_key)
        vocab[len(vocab)] = new_byte

        words_to_delete = []
        new_words = {}
        for word in pretoken_counts:
            new_word = []
            counter = 0
            while counter < len(word):
                if (
                    counter + 1 < len(word)
                    and (word[counter], word[counter + 1]) == max_pair_key
                ):
                    # we modify this word
                    new_word.append(new_byte)
                    counter += 2
                else:
                    new_word.append(word[counter])
                    counter += 1

            new_word = tuple(new_word)
            if new_word != word:
                new_words[new_word] = pretoken_counts[word]
                words_to_delete.append(word)

        for word in words_to_delete:
            del pretoken_counts[word]

        pretoken_counts.update(new_words)

        # todo: optimize candidate pairs?
    return vocab, merges
