from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def encode_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = raw_data.select_dtypes(include=["object", "category"]).columns
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded = encoder.fit_transform(raw_data[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))

    numeric = raw_data.drop(columns=categorical_cols).reset_index(drop=True)
    encoded_final = pd.concat([numeric.reset_index(drop=True), encoded_df], axis=1)
    return encoded_final