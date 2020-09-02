# Acknowledgments

We are indebted to the authors of [Screaming Channel](https://github.com/eurecom-s3/screaming_channels) who generously shared the code required to setup experiments and also grateful to Zihao Zhao for his help with configuring hardware for the experiments.

The scripts for training and testing are adopted from [ASCAD](https://github.com/ANSSI-FR/ASCAD).


# Trace Access
All traces used for training and testing are available from [here](https://drive.google.com/drive/folders/1RBjtpmGHaCutOn0fNcIsrVSY3RnPrpCv?usp=sharing).

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
│         │    ├── 100k_d1
│         │    |      └──100avg
│         │    ├── 100k_d2
│         │    |      └──100avg
│         │    ├── 100k_d3
│         │    |      └──100avg
│         │    ├── 100k_d4
│         │    |      └──100avg
│         │    ├── 100k_d5
│         │    |      └──100avg
│         │ 
│         │ 
└── for_testing
          ├── 3m
          │    └──10k_d6
          │          ├──100avg   
          │          └──1avg
          ├── 6m
          │    └──10k_d7
          │          ├──100avg   
          │          └──1avg          
          ├── 9m
          │    └──10k_d8
          │          ├──100avg   
          │          └──1avg          
          └── 15m
               └──10k_d9
                     ├──100avg   
                     └──1avg
```
