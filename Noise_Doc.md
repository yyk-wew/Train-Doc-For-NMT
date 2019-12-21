# Noise Generation
This work's idea is mainly based on Belinkov & Bisk [Synthetic and Natural Noise Both Break Neural Machine Translation](https://arxiv.org/abs/1711.02173) ICLR 2018 and [his repo](https://github.com/ybisk/charNMT-noise).

## Noise Collection
The natural noise is collected from lecture or corpus modification history, while other noise is synthetised manually. Here are noise script supports:
+ swap
+ key
+ middle
+ random
+ real/natural

## Settings
The script will read noise resource from file and add them into vanilia text according to the config param in config.yml, thus you need to install yaml.

This script has some modification for our experiments. A config file is like below:
```
file: [filepath1 (filepath2)]
lang: en                        # fr/de/cs/en
ftype: txt                      # filetype:sgm/txt
scrambling:   [real]            # swap, middle, random, key, real
distribution: [1]               # probability of running this op(0~1)
pair: false                     # true if need to pair two parallel dataset
```

Script will add noise into file1, if need to pair, it will add raw clean string copy as a placeholder into file2 if necessary.

## API
+ iterate_through(line):
  + this function will do the modification to each word in a sentence, and return a list whose element is a modified sentence.

+ swap(w, probability):
  + swap character in word according to the input probabilty.
