
# Exoplanet Observation Visibility Predictor

This code is designed to efficiently predict and plan large numbers of exoplanet transit and emission-phase observations simultaneously. It is intended primarily for large-scale survey planning at ground-based, high-resolution spectroscopic observatories. 

## Features

- **Batch observation planning:** Predict visibility windows for multiple exoplanet observations in a single run.
- **Airmass constraints:** Filter observations based on specified airmass limits.
- **Lunar constraints:** Account for Moon illumination when selecting viable observing windows.
- **Flexible observation coverage:** Add extra observing hours before or after an observation to extend the coverage window.

## Disclaimer

Although this code is designed to provide reliable predictions of observation visibility, always independently verify observing windows using the NASA Transit and Ephemeris Service before scheduling or conducting observations.

Actual observability may depend on additional instrument, telescope, and observing-condition constraints that are not captured by this code.


The code uses astropy's location so the supported observatory names can be found here:
https://github.com/astropy/astropy-data/blob/gh-pages/coordinates/sites.json?utm_source=chatgpt.com
