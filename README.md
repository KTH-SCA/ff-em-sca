# Far Field EM Side-Channel Analysis Project

This page gives information necessary to reproduce the article "Far Field EM Side-Channel Attack on AES Using Deep Learning", Ruize Wang, Huanyu Wang, Elena Dubrova, in 4th ACM Workshop on Attacks and Solutions in Hardware Security (ASHES’20), November 13, 2020, Orlando, USA,  available at eprint.


## Trace Access
All traces used for training and testing are available from [here](https://drive.google.com/drive/folders/1RBjtpmGHaCutOn0fNcIsrVSY3RnPrpCv?usp=sharing).

There are 7 files in the folder 'x'avg, where 'x' is 1, 100 or 1000. All these files are in ndarray format.

* *key.npy*: The 128 bits AES secret key.
* *pt.npy*: Numpy array stacked by 128 bits plaintext.
* *traces.npy*: Numpy array stacked by single trace with 400 sampling points.
* *label_0.npy*: 1-D numpy array generated by the first byte of Sbox output in the first round.
* *keylist.npy*: Numpy array stacked by 128 bits AES secret key, this file is used in ChipWhisperer Analyzer.
* *nor_traces_maxmin.npy*: Normalized traces file using min-max scaling.
* *nor_traces_meanstd.npy*: Normalized traces file using standardization.

The structure of folders are shown in the following:

```bash
5Mhz
├── for_training
│         ├── 1m
│         │    ├── 20k_d1
|         |    |      └──100avg
│         │    ├── 20k_d2
|         |    |      └──100avg
│         │    ├── 20k_d3
|         |    |      └──100avg
│         │    ├── 20k_d4
|         |    |      └──100avg
│         │    ├── 20k_d5
|         |    |      └──100avg
│         │    └── 100k_d1
│         │           └──100avg
│         ├── 2m
│         │    ├── 20k_d1
|         |    |      └──100avg
│         │    ├── 20k_d2
|         |    |      └──100avg
│         │    ├── 20k_d3
|         |    |      └──100avg
│         │    ├── 20k_d4
|         |    |      └──100avg
│         │    ├── 20k_d5
|         |    |      └──100avg
│         │    └── 100k_d1
│         │           └──100avg
│         ├── 4m
│         │    ├── 20k_d1
|         |    |      └──100avg
│         │    ├── 20k_d2
|         |    |      └──100avg
│         │    ├── 20k_d3
|         |    |      └──100avg
│         │    ├── 20k_d4
|         |    |      └──100avg
│         │    ├── 20k_d5
|         |    |      └──100avg
│         │    └── 100k_d1
│         │           └──100avg
│         ├── 8m
│         │    ├── 20k_d1
|         |    |      └──100avg
│         │    ├── 20k_d2
|         |    |      └──100avg
│         │    ├── 20k_d3
|         |    |      └──100avg
│         │    ├── 20k_d4
|         |    |      └──100avg
│         │    ├── 20k_d5
|         |    |      └──100avg
│         │    └── 100k_d1
│         │           └──100avg
│         └── cable
│              ├── 20k_d1
|              |      └──100avg
│              ├── 20k_d2
|              |      └──100avg
│              ├── 20k_d3
|              |      └──100avg
│              ├── 20k_d4
|              |      └──100avg
│              ├── 20k_d5
|              |      └──100avg
│              ├── 100k_d1
│              |      └──100avg
│              ├── 100k_d2
│              |      └──100avg
│              ├── 100k_d3
│              |      └──100avg
│              ├── 100k_d4
│              |      └──100avg
│              └── 100k_d5
│                     └──100avg
│           
│           
└── for_testing
          │ 
          ├── 3m
          │    └──10k_d6
          │          ├──1avg   
          │          ├──100avg             
          │          └──1000avg
          ├── 6m
          │    └──10k_d7
          │          ├──1avg   
          │          ├──100avg             
          │          └──1000avg       
          ├── 9m
          │    └──10k_d8
          │          ├──1avg   
          │          ├──100avg             
          │          └──1000avg    
          └── 15m
               └──10k_d9
                     ├──1avg   
                     ├──100avg             
                     └──1000avg
```


## Acknowledgments

We are indebted to the authors of [Screaming Channel](https://github.com/eurecom-s3/screaming_channels) who generously shared the code required to setup experiments and also grateful to the KTH students Martin Brisfors for his valuable advice on training deep learning models and Zihao Zhao for his help with configuring hardware for the experiments. The scripts for training and testing are adopted from [ASCAD](https://github.com/ANSSI-FR/ASCAD).
