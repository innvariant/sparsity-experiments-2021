The code in this folder is used to perform pruning and random prior experiments.

1. Folder **`dataset`**:

This folder contains the entire Reber grammar dataset used to perform experiments.
It also includes the split version of this dataset, titled [`train_data.csv`](https://github.com/innvariant/sparsity-experiments-2021/blob/master/rnn/dataset/train_data.csv) and [`test_data.csv`](https://github.com/innvariant/sparsity-experiments-2021/blob/master/rnn/dataset/test_data.csv).

Python script [`dataset.py`](https://github.com/innvariant/sparsity-experiments-2021/blob/master/rnn/dataset/dataset.py) is used to plot visualizations of this dataset.

2. Folder **`experiments`**:

This folder contains experiments conducted on the Reber grammar dataset, their results, plot visualizations, and saved state dictionaries and Random Graphs.

3. **`layer.py`**:

This Python script contains the implementation of various base recurrent layers, i.e., [`RNN_Tanh`](https://github.com/innvariant/sparsity-experiments-2021/blob/master/rnn/layer.py#L74), [`RNN_ReLU`](https://github.com/innvariant/sparsity-experiments-2021/blob/master/rnn/layer.py#L77), [`LSTM`](https://github.com/innvariant/sparsity-experiments-2021/blob/master/rnn/layer.py#L91), and [`GRU`](https://github.com/innvariant/sparsity-experiments-2021/blob/master/rnn/layer.py#L80). 

4. **`sparse.py`**:

This Python script contains the implementation of pruning recurrent networks ([`PruneRNN`](https://github.com/innvariant/sparsity-experiments-2021/blob/master/rnn/sparse.py#L8-L111)) and random prior experiments ([`ArbitraryStructureRNN`](https://github.com/innvariant/sparsity-experiments-2021/blob/master/rnn/sparse.py#L114-L216)) modules.