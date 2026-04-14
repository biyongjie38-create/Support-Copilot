from eval.run_eval import serialize_ragas_result


class _FakePandasFrame:
    def to_dict(self, orient=None):
        assert orient == "records"
        return [{"faithfulness": 0.91, "factual_correctness": 0.88}]


class _FakeRagasResult:
    def keys(self):
        return [0]

    def __getitem__(self, key):
        raise KeyError(key)

    def to_pandas(self):
        return _FakePandasFrame()


def test_serialize_ragas_result_falls_back_when_dict_conversion_raises_keyerror():
    result = serialize_ragas_result(_FakeRagasResult())

    assert result == {"scores": [{"faithfulness": 0.91, "factual_correctness": 0.88}]}
