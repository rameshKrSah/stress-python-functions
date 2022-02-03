# Functions used in stress research project

* `stress_models.py` - Functions to create CNN models for binary supervised and transfer stress classification. The hyper-parameters were found with grid search.
* `pretext_tasks.py` - Pre-text functions from research papers for self-supervised stress classification using time-series sensor data.
* `encoder_model.py` - Functions to create encoder and pre-text CNN models for self-supervised binary stress classification using time-series sensor data.
* `filters.py` - Functions for High and Low pass Butterworth filters for EDA data.
* `data_proc_wesad.py` - Functions for processing WESAD dataset.
* `data_proc_adarp.py` - Functions for processing ADARP dataset.
* `data_loader.py` - Functions to load WESAD and ADARP datasets.
