import itertools
import tensorflow_datasets as tfds

def gen_train(ds, num_val):
    found = {k: 0 for k in range(10)}
    for x, y in ds:
        if found[y.numpy()] < num_val:
            found[y.numpy()] += 1
        else:
            yield x, y

def gen_val(ds, num_val):
    done = 0
    found = {k: 0 for k in range(10)}
    for x, y in ds:
        if found[y.numpy()] < num_val:
            found[y.numpy()] += 1
            if found[y.numpy()] == num_val:
                done += 1
            yield x, y
        if done == 10:
            break

def make_data(num_val_train=400, num_val_extra=200):
    read_config = tfds.ReadConfig()
    read_config.options.experimental_deterministic = False
    train_ds = tfds.load("svhn_cropped", split="train", as_supervised=True)
    test_ds = tfds.load("svhn_cropped", split="test", as_supervised=True)
    extra_ds = tfds.load("svhn_cropped", split="extra", shuffle_files=True,
                         as_supervised=True, read_config=read_config)
    return (
        itertools.chain(gen_train(train_ds, num_val_train),
                        gen_train(extra_ds, num_val_extra)),
        itertools.chain(gen_val(train_ds, num_val_train),
                        gen_val(extra_ds, num_val_extra)),
        test_ds
    )

# for x, y in make_data()[1]:
#     pass
