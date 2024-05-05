# Multihead Diversity Analysis

This repository contains tools and data for analyzing the diversity metrics between individual heads in neural network models.

## Repository Contents

### Metrics Data

The `Metrics Data` folder contains:
- **JSON Data Files**: Contain diversity metrics for different model configurations, showing the diversity metrics between individual heads.
- **CSV Files**: Contain metric averages per layer.

### App

The `App` folder contains a Flask app that can be run to display the inter-head output disagreement metrics for each unique head combination over batch losses.

### Multihead_Diversity.ipynb

The `Multihead_Diversity.ipynb` notebook contains:
- **Data Extraction**: Extracts diversity metrics from different model configurations.
- **Visualization Creation**: Generates new visualizations or extracts data from new configurations.

It can be used to generate data for new model configurations.
