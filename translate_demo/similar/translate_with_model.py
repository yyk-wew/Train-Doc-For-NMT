# !_*_ coding:utf-8 _*_

import os


def translate(input, language):
    basis_path = os.getcwd()

    input_file_path = basis_path + '/' + 'input.txt'
    output_file_path = basis_path + '/' + 'output.txt'

    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    with open(input_file_path, 'w') as f:
        f.write(input)

    val = os.system('python -m nmt.nmt \
                        --out_dir=./enzh_gnmt_singleword \
                        --inference_input_file=./input.txt \
                        --inference_output_file=./output.txt')

    with open(output_file_path, 'r') as f:
        output = f.read()

    return output


if __name__ == "__main__":
    res = translate("I'm coming home.", "wo wo tou yi kuai qian si ge, hey hey")
    print(res)
