# Post-cafa Data Generation

Codes for generating the in-house data are in the subdirectory `in-house_data`. There, run `generate_data.sh` to generate the in-house-data.

Similarly, in subdirectory `cafa3_data`, run `generate_cafa3_data.sh` to generate the CAFA3 data.

Finally, in subdirectory `stacking_data`, run `generate_stacking_data.sh`.

Currently, the input data files (`ipscan`, `taxonomy`, `targetp`..) are located in `/data/henri`.

**NOTE:** The CAFA3 CC training data was later updated because there were some sequences missing. Therefore, we generated data matrices for the missing CC data separately, by removing any size-based filter operations (set all `min_count` parameters to zero in `load_data.py`) from the data generating code and changing `data_processsing.py` script's `process_taxonomy_line` function's `feature` field selection from 2 to 3. We combined the new data with the old data using the script:

```
DATA_DIRECTORY_PATH/combined_cafa3_cc/combine.py
```

The data directory for the new data (including the base `ipscan`, `taxonomy`.. files) is located here:

```
kronos:/data/henri/CC_new_CAFA3_data
```

**TODO:** Collect all input data files for in-house data and CAFA data into a directory that can be added to the project web page with these codes.

