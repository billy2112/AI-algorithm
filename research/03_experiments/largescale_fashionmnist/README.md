# Fashion-MNIST scale-up attempt (PCA features) - EXPLORATORY/WEAK

**Status:** FAILED/IMPROVISED FALLBACK - NOT VALID COMPARISON

- Attempted to load Fashion-MNIST via sklearn.datasets.fetch_openml('Fashion-MNIST') 
- OpenML fetch failed due to missing pandas / parser issues in this environment
- Script `scripts/run_largescale_fashionmnist.py` had fallback paths but they were not properly implemented; results.json contains errors
- **No valid subsampled/PCA'd dataset was successfully loaded for all baselines**
- **Cannot confirm identical data across HSAM/SLDA/GNG-U/replay baselines**
- **Variance retained unknown, sample counts unknown**

This result must be treated as **exploratory/invalid** and should **NOT** carry evidentiary weight. The digits results (8x8, 1797 images) remain the primary evidence.

## Update
- Installed pandas 2.2.3; attempted fetch_openml('Fashion-MNIST') again - still failed in this environment
- No working data loader available (torchvision/tensorflow not present; OpenML fetch blocked)
- **STOPPING** - no improvised fallback will be created. Scale-up results remain unavailable/invalid.
