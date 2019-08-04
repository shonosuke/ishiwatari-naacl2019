# ishiwatari-naacl2019

* Paper: [Learning to Describe Unknown Phrases with Local and Global Contexts](https://www.aclweb.org/anthology/N19-1350)
* Authors: [Shonosuke Ishiwatari](http://shonosuke.jp), [Hiroaki Hayashi](https://hiroakih.me/), [Naoki Yoshinaga](http://www.tkl.iis.u-tokyo.ac.jp/~ynaga/index.en.html), [Graham Neubig](http://www.phontron.com), [Shoetsu Sato](http://www.tkl.iis.u-tokyo.ac.jp/~shoetsu/index.en.html), [Masashi Toyoda](http://www.tkl.iis.u-tokyo.ac.jp/~toyoda/index_e.html), and [Masaru Kitsuregawa](http://www.tkl.iis.u-tokyo.ac.jp/Kilab/Members/memo/kitsure_e.html)

## Requirements
* Pytorch (1.0.0)
* torchtext (0.2.3)
* torchvision (0.1.9)
* numpy (1.15.4)
* nltk (3.2.5)

## Download Dataset
```bash
wget http://www.tkl.iis.u-tokyo.ac.jp/~ishiwatari/naacl_data.zip
unzip naacl_data.zip
```

## Run Experiments
```bash
cd exp
./run_oxford.sh  # Run expreiments on Oxford Dictionary [Gadetsky+ 2018]
./run_slang.sh   # Run experiments on Urban Dictionay [Ni & Wang 2017]
./run_wiki.sh    # Run experiments on Wikipedia Dataset [Ishiwatari+ 2019]
./run_wordnet.sh # Run experiments on Wordnet [Noraset+ 2017]
```

## Reproduce BLEU Scores in the paper
```bash
# After training
./test.sh
```

## Contact
Shonosuke Ishiwatari

ishiwatari (at) tkl.iis.u-tokyo.ac.jp