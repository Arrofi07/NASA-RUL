from sklearn.preprocessing import StandardScaler
import numpy as np

def build_features(train_data, test_data):
    train = train_data.copy()
    test = test_data.copy()

    # drop low variance sensors
    sensor_cols = [c for c in train.columns if "sensor" in c]

    variance = train[sensor_cols].var()
    low_var = variance[variance < 1e-3].index
    print(f"sensors with low variance:\n{variance[variance < 1e-3]}")

    # normalize by engine operation condition
    scaler = StandardScaler()

    train[sensor_cols] = scaler.fit_transform(train[sensor_cols])
    test[sensor_cols] = scaler.transform(test[sensor_cols])

    train.drop(columns=low_var, inplace=True)
    test.drop(columns=low_var, inplace=True)

    # rolling window features
    sensor_cols = [c for c in train.columns if "sensor" in c]

    window = 10

    for col in sensor_cols:
        train[f"{col}_mean"] = (
            train.groupby("engine_id")[col]
            .rolling(window)
            .mean()
            .reset_index(0, drop=True)
        )

        test[f"{col}_mean"] = (
            test.groupby("engine_id")[col]
            .rolling(window)
            .mean()
            .reset_index(0, drop=True)
        )

    for col in sensor_cols:
        train[f"{col}_std"] = (
            train.groupby("engine_id")[col]
            .rolling(window)
            .std()
            .reset_index(0, drop=True)
        )

        test[f"{col}_std"] = (
            test.groupby("engine_id")[col]
            .rolling(window)
            .std()
            .reset_index(0, drop=True)
        )

    for col in sensor_cols:
        train[f"{col}_diff"] = train.groupby("engine_id")[col].diff()
        test[f"{col}_diff"] = test.groupby("engine_id")[col].diff()

    # remove rows where rolling features are undefined only some columns
    feature_cols = [c for c in train.columns if "sensor" in c]

    train = train.dropna(subset=feature_cols)
    test = test.dropna(subset=feature_cols)

    return (train, test)



def create_sequences(train_data, seq_len=30):

    X_seq = []
    y_seq = []

    sensor_cols = [c for c in train_data.columns if "sensor" in c]

    for engine in train_data.engine_id.unique():
        engine_df = train_data[train_data.engine_id == engine]

        for i in range(len(engine_df) - seq_len):

            X_seq.append(engine_df.iloc[i:i+seq_len][sensor_cols].values)
            y_seq.append(engine_df.iloc[i+seq_len]["RUL"])

    return np.array(X_seq), np.array(y_seq)

def create_test_sequences(test_data, seq_len=30, sensor_cols=None):

    sequences = []

    if sensor_cols is None:
        sensor_cols = [c for c in test_data.columns if "sensor" in c]

    for engine_id in test_data["engine_id"].unique():
        engine = test_data[test_data["engine_id"] == engine_id]

        if len(engine) >= seq_len:
            seq = engine.iloc[-seq_len:][sensor_cols].values
            sequences.append(seq)

    return np.array(sequences)