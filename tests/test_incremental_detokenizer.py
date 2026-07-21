from auto_infer.serving.detokenizer import IncrementalTextDecoder


class PieceTokenizer:
    def __init__(self, pieces):
        self.pieces = pieces

    def convert_ids_to_tokens(self, token_ids, skip_special_tokens=True):
        return [self.pieces[token_id] for token_id in token_ids]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)


def test_stop_string_split_across_tokens_is_not_leaked():
    decoder = IncrementalTextDecoder(
        PieceTokenizer({1: "hello<", 2: "STOP>tail"}), stop=["<STOP>"]
    )

    first = decoder.push((1,))
    final = decoder.push((2,))

    assert first.text == "hello"
    assert not first.finished
    assert final.text == ""
    assert final.finished
    assert final.finish_reason == "stop"


def test_finish_flushes_partial_stop_prefix():
    decoder = IncrementalTextDecoder(
        PieceTokenizer({1: "answer<"}), stop=["<STOP>"]
    )

    assert decoder.push((1,)).text == "answer"
    final = decoder.finish("length")

    assert final.text == "<"
    assert final.finished
    assert final.finish_reason == "length"


def test_incomplete_unicode_fragment_is_held_until_stable():
    class ByteTokenizer(PieceTokenizer):
        def convert_tokens_to_string(self, tokens):
            text = "".join(tokens)
            return text.replace("<b1><b2>", "你").replace("<b1>", "�")

    decoder = IncrementalTextDecoder(
        ByteTokenizer({1: "A", 2: "<b1>", 3: "<b2>"}), stop=[]
    )

    assert decoder.push((1,)).text == "A"
    assert decoder.push((2,)).text == ""
    assert decoder.push((3,)).text == "你"


def test_decode_work_stays_bounded_as_output_grows():
    class CountingTokenizer:
        def __init__(self):
            self.maximum_decode_input = 0

        def decode(self, token_ids, skip_special_tokens=True):
            self.maximum_decode_input = max(
                self.maximum_decode_input, len(token_ids)
            )
            return "".join(chr(65 + token_id % 26) for token_id in token_ids)

    tokenizer = CountingTokenizer()
    decoder = IncrementalTextDecoder(tokenizer, stop=[])

    text = "".join(decoder.push((token,)).text for token in range(100))

    assert len(text) == 100
    assert tokenizer.maximum_decode_input <= 9


def test_push_after_terminal_is_ignored():
    decoder = IncrementalTextDecoder(PieceTokenizer({1: "x"}), stop=[])
    decoder.finish("length")

    late = decoder.push((1,))

    assert late.text == ""
    assert late.finished
    assert late.finish_reason == "length"
