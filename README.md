## TALL: Temporal Activity Localization via Language Query

This is the repository for our ICCV 2017 paper [_TALL: Temporal Activity Localization via Language Query_](https://arxiv.org/abs/1705.02101).

### Visual Features on TACoS
Download the C3D features for [training set](https://drive.google.com/file/d/1zQp0aYGFCm8PqqHOh4UtXfy2U3pJMBeu/view?usp=sharing)  and [test set](https://drive.google.com/file/d/1zC-UrspRf42Qiu5prQw4fQrbgLQfJN-P/view?usp=sharing) of TACoS dataset. Modify the path to feature folders in main.py

### Sentence Embeddings on TACoS
Download the Skip-thought sentence embeddings and sample files from [here](https://drive.google.com/file/d/1HF-hNFPvLrHwI5O7YvYKZWTeTxC5Mg1K/view?usp=sharing) of TACoS Dataset, and put them under exp_data folder.

### Reproduce the results on TACoS
`python main.py`

### Charades-STA
The sentence temporal annotations on Charades Dataset are available here: [train](https://drive.google.com/file/d/1ZjG7wJpPSMIBYnW7BAG2u9VVEoNvFm5c/view?usp=sharing), [test](https://drive.google.com/file/d/1QG4MXFkoj6JFU0YK5olTY75xTARKSW5e/view?usp=sharing). You may want to generate the skip-thought embeddings and C3D features on Charades-STA, and modify the codes slightly to reproduce the experiments.
