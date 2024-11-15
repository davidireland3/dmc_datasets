## Dataset Setup

1. Download the datasets from [here](https://warwickfiles.warwick.ac.uk/s/GrGH9RsyDRajASq?path=%2FDMC)
2. Extract them maintaining the following structure:
   ```
   /path/to/your/data/
   └── <bin_size>_bins/
       ├── finger-spin-expert
       ├── fish-swim-expert
       └── ...
   ```
3. Set the environment variable DMC_DISCRETE_DATA_DIR:
   ```bash
   export DMC_DISCRETE_DATA_DIR=/path/to/your/data
   ```
   Or pass the data_dir parameter directly to load_dataset().

## Dataset Usage

The datasets can be loaded as follows:
```python
from dmc_datasets.buffer_utils import load_dataset
dataset = load_dataset(task_name='finger', task='spin', bin_size=3, level='medium')
```
`dataset` will be a `ReplayBuffer` object (definition can be found in the code) which can be integrated into your training pipeline.

Currently we only support bin size of 3 for `cheetah-run`, `finger-spin`, `fish-swim`, `quaduped-walk`, `humanoid-stand`, `dog-trot` and bin sizes {10, 30, 50, 75, 100} for `dog-trot`. Each dataset has a `medium`, `expert`, `medium-expert` and `random-medium-expert` level.

## Environment wrappers

1. We also provide wrappers for the DMC dataset that be used to return gym environments with either factorised or atomic discrete action spaces:
   ```python
   from dmc_datasets.environment_utils import make_env

   env = make_env(task_name='finger', task='spin', bin_size=3, factorised=True)
   ```
   
2. This environment class can also be used to load a corresponding dataset:
   ```python
   from dmc_datasets.environment_utils import make_env

   env = make_env(task_name='finger', task='spin', bin_size=3, factorised=True)
   dataset = env.load_dataset(level='medium')
   ```
