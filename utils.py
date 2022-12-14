import commentjson


def read_model():
    with open('model.json', 'r') as handle:
        model = commentjson.load(handle)

    return model
