import pytest
import torch

from curated_transformers.repository.file import LocalFile
from curated_transformers.tokenizers import PiecesWithIds
from curated_transformers.tokenizers.legacy.xlmr_tokenizer import XLMRTokenizer

from ...compat import has_hf_transformers
from ...utils import torch_assertclose
from ..util import compare_tokenizer_outputs_with_hf_tokenizer


@pytest.fixture
def toy_tokenizer(test_dir):
    return XLMRTokenizer.from_files(
        model_file=LocalFile(path=test_dir / "toy.model"),
    )


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_from_hf_hub_equals_hf_tokenizer(sample_texts):
    compare_tokenizer_outputs_with_hf_tokenizer(
        sample_texts, "xlm-roberta-base", XLMRTokenizer
    )


def test_xlmr_toy_tokenizer(toy_tokenizer, short_sample_texts):
    encoding = toy_tokenizer(short_sample_texts)
    _check_toy_tokenizer(encoding)

    decoded = toy_tokenizer.decode(encoding.ids)
    assert decoded == [
        "I saw a girl with a telescope.",
        "Today we will eat pok ⁇  bowl, lots of it!",
        "Tokens which are unknown in ⁇  most ⁇  latin ⁇  alphabet ⁇  vocabularies.",
    ]


def _check_toy_tokenizer(pieces):
    assert isinstance(pieces, PiecesWithIds)
    assert len(pieces.ids) == 3
    assert len(pieces.pieces) == 3

    assert pieces.ids == [
        [0, 9, 466, 11, 948, 42, 11, 171, 169, 111, 29, 21, 144, 5, 2],
        [
            0,
            484,
            547,
            113,
            172,
            568,
            63,
            21,
            46,
            3,
            85,
            116,
            28,
            4,
            149,
            227,
            7,
            14,
            26,
            147,
            2,
        ],
        [
            0,
            484,
            46,
            95,
            7,
            140,
            123,
            222,
            46,
            25,
            116,
            25,
            20,
            3,
            637,
            3,
            149,
            77,
            54,
            3,
            11,
            28,
            30,
            53,
            19,
            66,
            16,
            15,
            3,
            8,
            84,
            21,
            29,
            19,
            66,
            232,
            50,
            458,
            5,
            2,
        ],
    ]

    assert pieces.pieces == [
        [
            "<s>",
            "▁I",
            "▁saw",
            "▁a",
            "▁girl",
            "▁with",
            "▁a",
            "▁t",
            "el",
            "es",
            "c",
            "o",
            "pe",
            ".",
            "</s>",
        ],
        [
            "<s>",
            "▁To",
            "day",
            "▁we",
            "▁will",
            "▁eat",
            "▁p",
            "o",
            "k",
            "é",
            "▁b",
            "ow",
            "l",
            ",",
            "▁l",
            "ot",
            "s",
            "▁of",
            "▁it",
            "!",
            "</s>",
        ],
        [
            "<s>",
            "▁To",
            "k",
            "en",
            "s",
            "▁which",
            "▁are",
            "▁un",
            "k",
            "n",
            "ow",
            "n",
            "▁in",
            "ペ",
            "▁most",
            "で",
            "▁l",
            "at",
            "in",
            "が",
            "▁a",
            "l",
            "p",
            "h",
            "a",
            "b",
            "e",
            "t",
            "際",
            "▁",
            "v",
            "o",
            "c",
            "a",
            "b",
            "ul",
            "ar",
            "ies",
            ".",
            "</s>",
        ],
    ]

    torch_assertclose(
        pieces.padded_tensor(padding_id=1),
        torch.tensor(
            [
                [
                    0,
                    9,
                    466,
                    11,
                    948,
                    42,
                    11,
                    171,
                    169,
                    111,
                    29,
                    21,
                    144,
                    5,
                    2,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    0,
                    484,
                    547,
                    113,
                    172,
                    568,
                    63,
                    21,
                    46,
                    3,
                    85,
                    116,
                    28,
                    4,
                    149,
                    227,
                    7,
                    14,
                    26,
                    147,
                    2,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    0,
                    484,
                    46,
                    95,
                    7,
                    140,
                    123,
                    222,
                    46,
                    25,
                    116,
                    25,
                    20,
                    3,
                    637,
                    3,
                    149,
                    77,
                    54,
                    3,
                    11,
                    28,
                    30,
                    53,
                    19,
                    66,
                    16,
                    15,
                    3,
                    8,
                    84,
                    21,
                    29,
                    19,
                    66,
                    232,
                    50,
                    458,
                    5,
                    2,
                ],
            ],
            dtype=torch.int32,
        ),
    )

    torch_assertclose(
        pieces.attention_mask().bool_mask.squeeze(dim=(1, 2)),
        torch.tensor(
            [
                [
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
                [
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
                [
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ],
            ]
        ),
    )
