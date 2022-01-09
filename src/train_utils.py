import pickle
import gin


@gin.configurable(module=__name__)
def get_training_hyperparams(num_epochs=50, batch_size=64):
    return num_epochs, batch_size


@gin.configurable(module=__name__)
def load_dataset(name="reverse_int_short"):

    # print(f"Loading dataset {name} ...")

    with open(f'datasets/{name}/src_train.ob', 'rb') as fp:
        src_train = pickle.load(fp)

    with open(f'datasets/{name}/tgt_train.ob', 'rb') as fp:
        tgt_train = pickle.load(fp)

    with open(f'datasets/{name}/src_test.ob', 'rb') as fp:
        src_test = pickle.load(fp)

    with open(f'datasets/{name}/tgt_test.ob', 'rb') as fp:
        tgt_test = pickle.load(fp)

    return (src_train, tgt_train), (src_test, tgt_test)
