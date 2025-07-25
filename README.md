# Generative modelling for fast image reconstruction and uncertainty quantification in astronomical imaging

This repository contains two novel image reconstruction methods based on the regularised conditional generative adversarial network (GAN) framework by [Bendel et al.](https://arxiv.org/abs/2210.13389). These methods are designed to quickly generate approximate posterior samples of the image from a set of noisy data, allowing for the creation of detailed image reconstructions with associated uncertainty maps. The two methods are:

**1. MMGAN**: *"Generative modelling for mass-mapping with fast uncertainty quantification"* [[arXiv]](https://arxiv.org/abs/2410.24197)

MMGAN is a novel mass-mapping method designed to quickly generate approximate posterior samples of the convergence field from shear data, MMGAN offers a fully data-driven approach to mass-mapping. These posterior samples allow for the creation of detailed convergence map reconstructions with associated uncertainty maps, making MMGAN a cutting-edge tool for cosmological analysis.

![MMGAN COSMOS convergence map reconstruction](/figures/MMGAN/cosmos_results.png)


**2. RI-GAN**: *"Generative imaging for radio interferometry with fast uncertainty quantification"* [in prep.]

RI-GAN is a novel radio interferometric imaging method that combines the regularised conditional GAN framework with model-based updates. This hybrid approach that is both based on the imaging model and data-driven, allows for fast generation of approximate posterior samples using the dirty image and PSF of the observation. This results in a fast imaging method that is robust to varying visibility coverages and which generalises well to unseen data, while providing informative uncertainty maps.

## Installation

After cloning the repository, if in a computing cluster, first run:
``` bash
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
```

To install the conda dependencies setting the correct channels:
``` bash
conda create --name cGAN --file conda_requirements.txt --channel pytorch --channel nvidia --channel conda-forge --channel defaults
```

Then activate the conda environment and install the pip requirements: 
``` bash
conda activate cGAN
pip install -r pypi_requirements.txt
```

## Reproducing our Results
### 
See ```docs/mass_mapping.md``` for detailed instructions on how to setup and reproduce the results from our paper on [MMGAN](https://arxiv.org/abs/2410.24197).

Alternatively, we have provided a [zenodo file](https://zenodo.org/records/14226221) with the weights of our trained model, as well as a number of simulations. 

Documentation for the RI-GAN method is currently in preparation, but we will provide a similar guide for reproducing the results from our paper on RI-GAN once it is ready. 


## Questions and Concerns
If you have any questions, or run into any issues, don't hesitate to reach out at jessica.whitney.22@ucl.ac.uk for the MMGAN method and academic@matthijsmars.com for the RI-GAN method. 

## References
This repository was forked from [rcGAN](https://github.com/matt-bendel/rcGAN) by [Bendel et al.](https://arxiv.org/abs/2210.13389), with significant changes and modification made by Whitney et al.


## Citation

If you find this code helpful, please cite our papers:

- **MMGAN:**
    ```
    @journal{2024arxiv,
      author = {Whitney, Jessica and Liaudat, Tobías and Price, Matthew and Mars, Matthijs and McEwen, Jason},
      title = {Generative modelling for mass-mapping with fast uncertainty quantification},
      year = {2024},
      journal={arXiv:2410.24197}
    }
    ```
- **RI-GAN:**
    ```
    @article{marsGenerativeImagingRadioInterferometry,
      author = {Mars, Matthijs and Liaudat, Tobías and Whitney, Jessica and McEwen, Jason},
      title = {Generative imaging for radio interferometry with fast uncertainty quantification},
      year = {},
      journal={in prep.}
    }

