# !_*_ coding:utf-8 _*_

f = open("train.zh", 'r', encoding='utf-8')

print("open")

content = f.read()
f.close()
print("read succeed")

vocab_dict = dict()
non_vocab_set = set()

for c in content:
    if not ('\u4e00' <= c <= '\u9fa5') :
        non_vocab_set.add(c)
    else:
        if c in vocab_dict:
            vocab_dict[c] = vocab_dict[c] + 1
        else:
            vocab_dict[c] = 1

print("Generate Dict")

with open("vocab.zh", 'w', encoding='utf-8') as f:
    for word in vocab_dict:
        if vocab_dict[word] >= 3:
            f.write(word + '\n')

print("finish")
