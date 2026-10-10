import numpy as np

from src.data import load_data
from src.features import build_preprocessor


def test_load_data_cleans_columns():
    df = load_data()
    assert "customerID" not in df.columns
    assert set(df["Churn"].unique()) == {0, 1}
    assert df["TotalCharges"].notna().all()


def test_preprocessor_has_no_nans():
    df = load_data()
    X = df.drop(columns="Churn")
    Xt = build_preprocessor(X).fit_transform(X)
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    assert Xt.shape[0] == len(X)
    assert not np.isnan(Xt).any()

    