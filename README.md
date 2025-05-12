# CombinedFeatureExtraction

Combined Feature Extraction is a collection of 3 pipelines for extracting and quantifying morpometrics from segmented masks. It contains the following plugins:

- Classical Features
 - Pathomic
 - Extended Clinical
- ExpandedGranularFeatures

At the high level using image analysis techniques, the plugin extracts image and contour-based pathomic features; namely, area, mesangial area for each glomerulus, average TBM tickness, average cell thickness and luminal fraction for each tubule and arterial area for each artery. The extended clinical features also include Area and Radius of each FTU.

The expanded granular plugin performs Sub-compartmentalization of Kidney microanatom ical structures. Namely, Nuclei, Eosinophilic and Lumen. Afterwards, it extracts the fallowing features: Morphological, Texture, Color and Distance transform.

