## TALL: Temporal Activity Localization via Language Query

This is the repository for our ICCV 2017 paper [_TALL: Temporal Activity Localization via Language Query_](https://arxiv.org/abs/1705.02101).

### Visual Features on TACoS
Download the C3D features for [training set](https://drive.google.com/file/d/1zQp0aYGFCm8PqqHOh4UtXfy2U3pJMBeu/view?usp=sharing)  and [test set](https://drive.google.com/file/d/1zC-UrspRf42Qiu5prQw4fQrbgLQfJN-P/view?usp=sharing) of TACoS dataset. Modify the path to feature folders in main.py

### Sentence Embeddings on TACoS
Download the Skip-thought sentence embeddings and sample files from [here](https://drive.google.com/file/d/1HF-hNFPvLrHwI5O7YvYKZWTeTxC5Mg1K/view?usp=sharing) of TACoS Dataset, and put them under exp_data folder.

### Reproduce the results on TACoS
`python main.py`

### Charades-STA anno download
The sentence temporal annotations on [Charades](http://allenai.org/plato/charades/) dataset are available here: [train](https://drive.google.com/file/d/1ZjG7wJpPSMIBYnW7BAG2u9VVEoNvFm5c/view?usp=sharing), [test](https://drive.google.com/file/d/1QG4MXFkoj6JFU0YK5olTY75xTARKSW5e/view?usp=sharing). The format is "[video name] [start time] [end time]##[sentence]". You may want to generate the skip-thought embeddings and C3D features on Charades-STA, and modify the codes slightly to reproduce the experiments.

### Updates on Charades-STA performance
I did some anno cleaning for Charades-STA (compared to the version I used in ICCV paper), the updated performance is listed below. Please compare to these results when using Charades-STA.

| Model            | R@1,IoU=0.5 | R@1,IoU=0.7 | R@5,IoU=0.5 | R@5,IoU=0.7 |
| :--------------- | ----------: | ----------: | ----------: | ----------: | 
| CTRL (aln)       |   17.69     |    5.91     |    55.54    |     23.79   |
| CTRL (reg-p)     |   19.22     |    6.64     |    57.98    |     25.22   |
| CTRL (reg-np)    |   21.42     |    7.15     |    59.11    |     26.91   |
