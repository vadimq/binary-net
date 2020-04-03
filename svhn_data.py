import itertools
import tensorflow_datasets as tfds

read_config = tfds.ReadConfig()
read_config.options.experimental_deterministic = False
train_ds = tfds.load("svhn_cropped", split="train", shuffle_files=True,
                     as_supervised=True, read_config=read_config)
extra_ds = tfds.load("svhn_cropped", split="extra", shuffle_files=True,
                     as_supervised=True, read_config=read_config)

num_val_train = 400
num_val_extra = 200

# <codecell>

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

def gen_train(ds, num_val):
    found = {k: 0 for k in range(10)}
    for x, y in ds:
        if found[y.numpy()] < num_val:
            found[y.numpy()] += 1
        else:
            yield x, y

# <codecell>

# for x, y in itertools.chain(gen_train(train_ds, num_val_train),
#                             gen_train(extra_ds, num_val_extra)):
#     pass

for x, y in itertools.chain(gen_val(train_ds, num_val_train),
                            gen_val(extra_ds, num_val_extra)):
    pass
