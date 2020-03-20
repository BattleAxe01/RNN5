def pred(x_test):
    import tensorflow as tf

    # load model
    from tensorflow.keras.models import load_model

    model = load_model("./model")

    predtiction = model.predict(x_test)
    print(predtiction)
    return predtiction
