# Notes on Papers
## Early Stage Deforestation Detection in the Tropics with L-Band SAR
### Problem
Cloud cover disturbs detection of deforestation, especially during rainy season.

### Procedure
1. Use PALSAR-2 to record HV and HH information.
2. Wack Calculations
    - Images are corrected for $\gamma^0$, which is the backscattering coefficient per area normal to incidence angle $\gamma^0 = \frac{\sigma^0}{cos\theta}$.
    - $\gamma^0_{HH}$ has been found to be useful for analysis with a threshold. Here it is used for early stage deforestation.
    - $\sigma^0$ varies with the seasons (specifically cited are changes from ice and snow, as well as rain. This somewhat undermines the original reason for this, but I guess they can calibrate).


## DETER-B
### Problem
PRODES shows a reduction in area sizes for forest clearing and existing systems are only able to detect areas in the range of 25ha - 100ha.

DETER-B measures clear-cut deforestation, deforestation with vegetation (reduced impact?) and mining, while PRODES measures clear-cut and residue(?)
### Procedure
1. Acquire Images
    - I think they are just padding the paper here.
2. Correct Geometry of Images
    - Based on nearest neighbor resampling and second degree polynomial interpolation (which one?)
    - Remove any areas significantly covered by clouds, etc.
3. Apply Linear Spectral Mixing Model
    - Example of a mixture model[^wikipedia-mixture-model]
    - Estimates the fraction of soil, vegetation, and shade in each image pixel.
    - Can be written as:
    $$
    d_i = \sum_{j = 1}^r{s_{ij}a_j + e_i}
    $$
    Where $d_i$ is the value for band $i$, $a_j$ is the fractional area covered by the $j$th component, $s_{ij}$ is the $i$th component of the vector for the $j$th mixture component (the vector $s_j$ is just the reflectance value for band $i$), and $e_i$ is the error term for the $i$th band.

    From this I gather that the vector $s_i$ is the reflectance (fraction of reflected light) for each component (soil, vegetation, and shade) in band $i$.

    The solution to this model is constrained by
    $$
    \sum_{j=1}^r{a_j} = 1, a_j \geq 0
    $$

    Solved for $a_j$ using constrained least squares and weighted least square methods.

    - The soil fraction highlights features of logging such as log decks[^log-decks-product-page], skid trails, roads, etc. In areas of reduced impact logging (where they keep some trees, I imagine), features are likely **only** detected over soil.

4. Retrieve Unobserved Areas
    - Generated on a bimonthly (twice per month or once every two months?) generated from cloud and shade vectors.
5. Generate PRODES Mask
    - Mask including all historical clear-cut deforestation as well as water.
    - Prevents repetition of previously deforested areas (ostensibly blocks out cleared areas already).
6. Visual Interpretation
    - Uses:
        - Soil fraction.
        - False color composites of AWIFS data.
        - Landsat, LISS[^liss-data-products], and DMC[^dcc-factsheet] time series.
    - Monthly mapping of:
        - Burn scars.
        - Regular / conventional selective logging.
7. Audit
    - Not interesting and did not read

## Using CNNs
### Problem:
No problem, just looks like they want to use CNNs.

### Procedure
Used three CNN architectures:
- UNet
    - Skip connections between upsampling and downsampling layers
- SharpMask
- ResUNet
    - Best overall model
    - Skip connections between adjacent layers

And compared against:
    - Random Forest
        - Hyperparameters of 500 trees and 3 random variables for splits.
    - MLP

Used PRODES data as ground truth masks.

### Results
- Found DL models generally outperformed traditional machine learning (with only the exception of precision over the 2017-2018 timeframe)
- DL models require less postprocessing of images to remove noise / improve results.
- Highlight the interesting fact that because deforestation is relatively rare, the change-no-change ratio is imbalanced. As a result there can be overfitting / a high bias in some accuracy metrics (it can be weirdly high on many maps.)

## Glossary of TLAs and ETLAs
- **AWIFS**: Advanced Wide Field Sensor[^awifs-technical-documentation]
    - Sensor seemingly hosted by NASA on a satellite.
    - Has worse spatial resolution than landsat (only 300m - 80m), but higher temporal resolution (5 days).
    - Collects information in four bands
- **BLA**: Brazilian Legal Amazon
    - Largest global rainforest on earth; containing almost 30% of all rainforests.
- **DEGRAD**: Brazilian Amazon Forest Degradation Project
- **DETER**: Near Real-Time Deforestation Detection
    - Created by INPE, uses daily temporal resolution MODIS 250m data.
- **DETEX**: Selective Logging Detection Project
- **DMC**: Deisaster Management Constellation
    - International collaborative constellation of satellites for mitigating natural disasters at a short timescale.
- **IBAMA**: Brazilian Institute of Environmental and Renewable Natural Resources
- **INPE**: National Institute for Space Research
    - Responsible for deforestation surveillance.
- **INPE-CRA**: Amazon Regional Center of INPE
- **LISS**: High Resolution Multi-Spectral Camera
    - Operated for NASA earth data?
- **MODIS**: Moderate Resolution Imaging Spectrodiometer
    - Images hosted by NASA?[^modis-nasa-dataset]
- **PALSAR-2**: Phased Array type L-band Synthetic Aperture Radar
    - System for measurement using SAR
    - Receives HV polarized waves with a resolution of 1degree and a spatial resolution of 50m
- **PPCDAM**: Federal Action Plan for Prevention and Control of Deforestation in the Amazon
- **PRODES**: Amazon Deforestation Monitoring Project
    - Collects official deforestation data.
    - Uses Landsat class satellite imagery at 20m-30m resolution and a 16 day revisit rate.
- **SAR**: Synthetic Aperture Radar
    - Available under all weather conditions (including cloud cover).

## Miscellaneous Notes
Clear-cut deforestation is different from reduced-impact logging. Clear-cut is just destroying everything, while reduced impact is[^reduced-impact-logging-document]

The terraclass project[^terraclass-project-flowchart] is used to map land use in the Amazon.

Light polarization[^polarimetry-nasa-tutorial] is determined by its electric component. Linearly oriented structures tend to preserve / reflect polarization and preserve coherence, while randomly oriented structures can scatter the signal and depolarize light. Types of polarimetry include:
    - HH, which transmits and receives horizontal waves
    - HV, which transmits horizontal waves and receives vertical waves
    - VV, which transmits and receives vertical waves
    - VH, which transmits vertical waves and receives horizontal waves

Dual-polarization systems might transmit in one polarization but receive in two. Quad-pol also exists by alternating H and V transmission and receiving in dual-pole mode, but requires higher accuracy.

Looks like everything goes back to PRODES; can't tell if this is because it's better, it's official, or both.

[^awifs-technical-documentation]: https://ntrs.nasa.gov/api/citations/20070038247/downloads/20070038247.pdf
[^dcc-factsheet]: https://learningzone.rspsoc.org.uk/index.php/Datasets/DMC/Key-Facts-DMC
[^liss-data-products]: https://cmr.earthdata.nasa.gov/search/concepts/C1214621700-SCIOPS
[^log-decks-product-page]: https://uniforest.com/product/log-decks
[^modis-nasa-dataset]: https://modis.gsfc.nasa.gov/
[^polarimetry-nasa-tutorial]: https://nisar.jpl.nasa.gov/mission/get-to-know-sar/polarimetry/
[^reduced-impact-logging-document]: https://pdf.usaid.gov/pdf_docs/Pnact287.pdf
[^terraclass-project-flowchart]: https://www.researchgate.net/figure/TerraClass-project-flowchart_fig4_304307507
[^wikipedia-mixture-model]: https://en.wikipedia.org/wiki/Mixture_model

