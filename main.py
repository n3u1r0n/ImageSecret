import fire
import bitarray
from PIL import Image
import numpy as np

DATALENGTHSIZE = 32
NAMELENGTHSIZE = 8
MOD = 8

def convert_block_up(array, blocksize=8, dtype=int):
    array = array
    result = np.zeros(len(array) * blocksize, dtype=dtype)
    for i in reversed(range(blocksize)):
        result[i::blocksize] = array % 2
        array //= 2
    return result.astype(dtype)

def convert_block_down(array, blocksize=8, dtype=int):
    array = np.append([0] * ((blocksize - len(array)) % blocksize),
                      array).astype(dtype)
    result = np.zeros(len(array) // blocksize, dtype=dtype)
    for i in range(blocksize):
        result *= 2
        result += array[i::blocksize]
    return result.astype(dtype)

def file_to_data(file_name):
    with open(file_name, 'rb') as file:
        data = np.frombuffer(file.read(), dtype=np.uint8).copy()
    return convert_block_up(data, dtype=bool)

def data_to_file(data, file_name):
    data = convert_block_down(data, dtype=np.uint8)
    with open(file_name, 'wb') as file:
        file.write(data)

def image_to_array(image_name):
    image = np.array(Image.open(image_name), dtype=np.uint8)
    return image.flatten(), image.shape

def array_to_image(array, shape, image_name):
    Image.fromarray(array.reshape(shape)).save(image_name)

def encode(file_name, image_name, o='out', m=3):
    data = file_to_data(file_name)
    name = convert_block_up(np.frombuffer(bytes(file_name, 'utf-8'),
                                          dtype=np.uint8).copy(), dtype=bool)
    array, shape = image_to_array(image_name)
    data_length = len(data)
    name_length = len(name)
    available_bytes = len(array) - 1 - (DATALENGTHSIZE + NAMELENGTHSIZE + name_length) // MOD
    mod = (data_length - 1) // available_bytes + 1
    while mod > m and mod > 8:
        array = array.reshape(shape)
        shape = (shape[0] * 2, shape[1] * 2, shape[2])
        new_array = np.zeros(shape, dtype=np.uint8)
        new_array[0::2,0::2,:] = array.copy()
        new_array[0::2,1::2,:] = array.copy()
        new_array[1::2,0::2,:] = array.copy()
        new_array[1::2,1::2,:] = array.copy()
        array = new_array.flatten()
        available_bytes = len(array) - 1 - (DATALENGTHSIZE + NAMELENGTHSIZE + name_length) // MOD
        mod = (data_length - 1) // available_bytes + 1
    
    lower, upper = 0, 1
    array[lower:upper] -= (array[lower:upper] % pow(2, MOD)).astype(np.uint8)
    array[lower:upper] += mod - 1
    
    lower, upper = upper, upper + DATALENGTHSIZE // MOD
    array[lower:upper] -= (array[lower:upper] % pow(2, MOD)).astype(np.uint8)
    array[lower:upper] += \
        convert_block_down(convert_block_up(np.array([data_length]), DATALENGTHSIZE, np.uint8), MOD, np.uint8)

    lower, upper = upper, upper + NAMELENGTHSIZE // MOD
    array[lower:upper] -= (array[lower:upper] % pow(2, MOD)).astype(np.uint8)
    array[lower:upper] += \
        convert_block_down(convert_block_up(np.array([name_length]), NAMELENGTHSIZE, np.uint8), MOD, np.uint8)

    lower, upper = upper, upper + name_length // MOD
    array[lower:upper] -= (array[lower:upper] % pow(2, MOD)).astype(np.uint8)
    array[lower:upper] += convert_block_down(name, MOD, np.uint8)

    lower, upper = upper, upper + (data_length - 1) // mod + 1
    array[lower:upper] -= (array[lower:upper] % pow(2, mod)).astype(np.uint8)
    array[lower:upper] += convert_block_down(data, mod, np.uint8)

    array_to_image(array, shape, '{}.png'.format(o))

def decode(image_name):
    array, _ = image_to_array(image_name)

    lower, upper = 0, 1
    mod = (array[lower:upper] % pow(2, MOD))[0] + 1

    lower, upper = upper, upper + DATALENGTHSIZE // MOD
    data_length = convert_block_down(convert_block_up(array[lower:upper] % pow(2, MOD), MOD), DATALENGTHSIZE)[0]

    lower, upper = upper, upper + NAMELENGTHSIZE // MOD
    name_length = convert_block_down(convert_block_up(array[lower:upper] % pow(2, MOD), MOD), NAMELENGTHSIZE)[0]

    lower, upper = upper, upper + name_length // MOD
    file_name = convert_block_down(convert_block_up(array[lower:upper] % pow(2, MOD), MOD), dtype=np.uint8).tobytes().decode('utf-8')
    
    lower, upper = upper, upper + (data_length - 1) // mod + 1
    data = convert_block_up(array[lower:upper] % pow(2, mod), mod, bool)

    data_to_file(data, file_name)

if __name__ == '__main__':
    fire.Fire({
        'encode': encode,
        'decode': decode
    })