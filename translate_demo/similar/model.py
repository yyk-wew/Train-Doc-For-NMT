#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os

# def translate(code1, lang):
#     L = ['English']
#     code2 = code1 + L[lang] + '翻译结果'
#     return code2


def translate(input, language):
    basis_path = os.getcwd()

    input_file_path = basis_path + '/' + 'input.txt'
    output_file_path = basis_path + '/' + 'output.txt'

    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    with open(input_file_path, 'w') as f:
        f.write(input)

    os.system("subword-nmt apply-bpe -i input.txt -c code.file -o input_code.txt")
    # 这里万一炸了catch一下

    val = os.system('python -m nmt.nmt \
                        --out_dir=./enzh_gnmt_singleword \
                        --inference_input_file=./input_code.txt \
                        --inference_output_file=./output.txt')

    # 这里根据val的值可以判断有没有执行成功，炸了也catch一下

    with open(output_file_path, 'r') as f:
        output = f.read()
        output = output.replace(" ","")
    return output
