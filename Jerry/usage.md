## Number of Trials for Maximum Exploration

The current parameter grid defined has a very large number of potential combinations:
- 3 batch sizes (16, 32, 64)
- 2 epoch options (30, 50)
- 3 learning rates (0.001, 0.0005, 0.0001)
- 3 optimizers (adam, sgd, rmsprop)
- 3 convolutional dropout rates (0.1, 0.25, 0.4)
- 3 dense dropout rates (0.3, 0.5, 0.7)
- 2 convolutional filter configurations ((32, 64, 128), (64, 128, 256))
- 2 kernel sizes (3, 5)
- 3 dense unit options (256, 512, 1024)
- 2 batch normalization options (True, False)
- 2 patience options (5, 10)
- 2 rotation range options (10, 20)
- 2 width shift range options (0.1, 0.2)
- 2 height shift range options (0.1, 0.2)
- 2 shear range options (0.1, 0.2)
- 2 zoom range options (0.1, 0.2)
- 1 horizontal flip option (True)

Multiplying all these options gives: 3×2×3×3×3×3×2×2×3×2×2×2×2×2×2×2×1 = 663,552 possible combinations.

Testing all combinations would be impractical, therefore:

1. **For thorough exploration**: Run 50-100 trials using random search (`--max_trials 100 --random_search`)
2. **For balanced approach**: Run 20-30 trials (`--max_trials 30`)
3. **For quick results**: Stick with the default 10 trials

The script is designed to automatically switch to random search if the number of combinations exceeds the max_trials parameter, so it will work efficiently regardless of what we choose.

## Default Output Directory

The default output directory structure in the script is:
```
<project_root>/grid_search_results/grid_search_YYYYMMDD_HHMMSS/
```

Where:
- `<project_root>` is the parent directory of the script's directory
- `grid_search_results` is the default subfolder name (which can be changed using the `--output_dir` parameter)
- `grid_search_YYYYMMDD_HHMMSS` is a timestamped folder created for each grid search run

For example, if we run the script without changing any parameters, the results might be saved to:
```
/path/to/your/project/grid_search_results/grid_search_20250517_123456/
```

Inside this directory, we'll find:
- A subfolder for each trial (trial_000, trial_001, etc.)
- Summary files (grid_search_results.csv, parameter_importance.png, etc.)
- The best model and its parameters

If we want to specify a different output directory, we can use:
```
python grid_search.py --output_dir my_custom_results
```

This would save results to:
```
/path/to/your/project/my_custom_results/grid_search_YYYYMMDD_HHMMSS/
```