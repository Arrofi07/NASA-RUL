
def preprocess(train_data, test_data, cols, clip=False):
    # load and rename columns
    train = train_data
    train.columns = cols

    test = test_data
    test.columns = cols

    # compute RUL
    max_cycle = train.groupby("engine_id")["cycle"].max()

    train["RUL"] = train.apply(
        lambda row: max_cycle[row.engine_id] - row.cycle,
        axis=1
    )

    # optional clip to improves training
    if clip == True:
        train["RUL"] = train["RUL"].clip(upper=125)

    return (train, test)