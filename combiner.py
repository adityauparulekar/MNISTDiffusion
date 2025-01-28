import os, random, sys, shutil

p = 0.5
end_size = 50000
for i in random.sample(range(60000), int(p*end_size)):
    if i % 1000 == 0:
        print(i)
    shutil.copy(f'datasets/fmnist_images/image_{i}.png', f'datasets/combo_images/image_f{i}.png')
for i in random.sample(range(60000), end_size - int(p*end_size)):
    if i % 1000 == 0:
        print(i)
    shutil.copy(f'datasets/mnist_images/image_{i}.png', f'datasets/combo_images/image_m{i}.png')