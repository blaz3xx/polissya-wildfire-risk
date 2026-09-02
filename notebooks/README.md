# Notebook

[`polissya_wildfire_risk.ipynb`](polissya_wildfire_risk.ipynb) is the full pipeline: data collection, EDA, feature engineering, modelling, and live inference, phases 0 through 5. Markdown commentary is in Ukrainian; code, variable names, and outputs are in English.

## Size note

The file is about 11 MB because it keeps all rendered outputs, including 14 figures and an embedded Folium map. That is under GitHub's limits, but the in-browser notebook viewer can be slow or refuse to render files this size. Two reliable ways to read it:

- **Google Colab** — the badge at the top of the main README opens it directly.
- **nbviewer** — `https://nbviewer.org/github/blaz3xx/polissya-wildfire-risk/blob/main/notebooks/polissya_wildfire_risk.ipynb`

If you would rather keep the repository light, strip the outputs before committing:

```bash
pip install nbstripout
nbstripout notebooks/polissya_wildfire_risk.ipynb
```

That drops it to a few hundred kilobytes, at the cost of losing the stored figures. The figures used in the README are already saved separately in [`../assets/`](../assets/), so nothing is lost from the documentation.

## Requirements to run it

- `FIRMS_MAP_KEY` in Colab Secrets or as an environment variable ([free key](https://firms.modaps.eosdis.nasa.gov/api/map_key/))
- Everything else in [`../requirements.txt`](../requirements.txt)

First run downloads and caches roughly 100 API responses. Later runs read the cache.
