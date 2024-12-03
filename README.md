# Generative modelling for mass-mapping with fast uncertainty quantification [[arXiv]](https://arxiv.org/abs/2410.24197)

MMGAN is a novel mass-mapping method based on the regularised conditional generative adversarial network (GAN) framework by [Bendel et al.](https://arxiv.org/abs/2210.13389). Designed to quickly generate approximate posterior samples of the convergence field from shear data, MMGAN offers a fully data-driven approach to mass-mapping. These posterior samples allow for the creation of detailed convergence map reconstructions with associated uncertainty maps, making MMGAN a cutting-edge tool for cosmological analysis.

![MMGAN COSMOS convergence map reconstruction](/figures/MMGAN/cosmos_results.png)


## Setup
See ```docs/setup.md``` for basic environment setup instructions.

## Reproducing the our Results
### 
See ```docs/mass_mapping.md``` for instructions on how to setup and reproduce our COSMOS results.

## Questions and Concerns
If you have any questions, or run into any issues, don't hesitate to reach out at jessica.whitney.22@ucl.ac.uk

## References
This repository was forked from [rcGAN](https://github.com/matt-bendel/rcGAN) by [Bendel et al.](https://arxiv.org/abs/2210.13389), with significant changes and modification made by Whitney et al.


## Citation
If you find this code helpful, please cite our paper:
```
@journal{2024arxiv,
  author = {Whitney, Jessica and Liaudat, Tobías and Price, Matthew and Mars, Matthijs and McEwen, Jason},
  title = {Generative modelling for mass-mapping with fast uncertainty quantification},
  year = {2024},
  journal={arXiv:2410.24197}
}
```