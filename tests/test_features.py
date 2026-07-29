import numpy as np

from llm_integrity.features import FeatureExtractor, Standardizer, surface_features


def test_feature_shapes_and_finiteness():
    texts = ["答案是42。", "def f(x):\n    return x + 1"]
    rows = [{"expected_answer": "42"}, {"category": "code"}]
    values = FeatureExtractor(hashed_dimension=32).transform(texts, rows)
    assert values.shape == (2, 11 + 32 + 5)
    assert np.isfinite(values).all()
    standardized = Standardizer().fit_transform(values)
    assert np.isfinite(standardized).all()
