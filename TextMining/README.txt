# Text Mining: Italian Job Advertisements Analysis

## Execution Instructions

1. Ensure all required packages are installed:
```
   pip install -r requirements.txt
```

2. Place the dataset `sample_skill_extraction.csv` and tmes_functions.py in the project root directory.

3. Execute the notebook `text_mining_ita_jobs_ads.ipynb` from start to finish.

4. To visualize the outputs and plots it is possible to open the already created folder. 
There are html files regarding BERTopic and LDA, with their respective interactive interfaces, to open them you have to download.

**Note:** The notebook uses a caching system. To force regeneration of any component, set the corresponding flag to `True` in the configuration cell:
- `FORCE_REGENERATE_DATA`
- `FORCE_REGENERATE_EMBEDDINGS`
- `FORCE_REGENERATE_CLASSIFIERS`
- `FORCE_REGENERATE_TOPICS`

