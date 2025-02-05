# **tmmax: transfer matrix method with jax**

<div align="center">
  <a href="https://pypi.org/project/tmmax/">
    <img src="https://github.com/bahremsd/tmmax/blob/master/docs/images/logo_tmmax.png" alt="tmmax">
  </a>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#introduction">Introduction</a></li>
    <li><a href="#documentation">Documentation</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#database">Database</a></li>
    <li><a href="#benchmarks">Benchmarks</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#credits">Credits</a></li>
    <li><a href="#contact-and-support">Contact and Support</a></li>
  </ol>
</details>

## Introduction

`tmmax` is a high-performance computational library designed for efficient calculation of optical properties in multilayer thin-film structures. Engineered to serve as a Swiss Army knife tool for thin-film optics research, `tmmax` integrates a comprehensive suite of advanced numerical methods. At its core, `tmmax` leverages JAX to enable just-in-time (JIT) compilation, vectorized operations, and XLA (Accelerated Linear Algebra) optimizations, dramatically accelerating the evaluation of optical responses in multilayer coatings. By exploiting these capabilities, `tmmax` achieves exceptional computational speed, making it the optimal choice for modeling and analyzing complex systems.

Originally architected for CPU-based execution to ensure accessibility and scalability across diverse hardware configurations, `tmmax` seamlessly extends its computational efficiency to GPU and TPU platforms, thanks to JAX’s unified execution model. This adaptability ensures that high-performance simulations can be executed efficiently across a range of computing environments without modifying the core implementation. Moreover, `tmmax` natively supports automatic differentiation (AD) through JAX’s powerful autograd framework, allowing users to compute analytical gradients of optical properties with respect to arbitrary system parameters. This capability makes it particularly well-suited for gradient-based inverse design, machine learning-assisted optimization, and parameter estimation in photonic applications, providing a direct pathway to next-generation thin-film engineering.


## Documentation

The complete documentation for tmmax is available in the [Example Gallery](https://github.com/bahremsd/tmmax/tree/master/docs/examples) within the `docs` directory. This repository provides an extensive set of examples demonstrating the key functionalities of tmmax, enabling users to efficiently analyze and manipulate multilayer thin-film stacks.

The examples cover fundamental and advanced use cases, including:

- Material Database Management: Retrieving wavelength-dependent refractive index (n) and extinction coefficient (k) data from the built-in material database. Users can seamlessly add new materials, modify existing entries, or remove materials while maintaining database integrity.

- Thin-Film Optical Properties: Computing reflection (R), transmission (T), and absorption (A) spectra for both coherent and incoherent multilayer thin-film structures. These calculations leverage the Transfer Matrix Method (TMM) for rigorous wave propagation analysis.

- Filter Design and Optimization: Rapid simulation of optical filters, showcasing how tmmax efficiently models various thin-film coatings, such as anti-reflective coatings, high-reflectivity mirrors, and bandpass filters.

Each example is designed to be highly modular, providing users with clear, structured workflows to integrate tmmax into their own research or engineering projects. Whether designing custom multilayer coatings or optimizing optical performance, the documentation serves as a comprehensive guide to leveraging the full computational power of tmmax.



## Usage

To compute the reflection and transmission spectra of a multilayer thin-film stack using the tmmax framework, consider the following example. Suppose we have a coherent multilayer structure consisting of [Air, Y₂O₃, TiO₂, Y₂O₃, TiO₂, Y₂O₃, TiO₂, SiO₂], where the incident wavelength varies from 500 nm to 700 nm, and the angle of incidence spans from 0° to 70°. The calculation is performed as follows:

```python
import jax.numpy as jnp
from tmmax.tmm import tmm

# Define your multilayer stack and simulation parameters

material_list = ["Air", "Y2O3", "TiO2", "Y2O3", "TiO2", "Y2O3", "TiO2", "SiO2"]
thickness_list = jnp.array([630e-9, 200e-9, 630e-9, 200e-9, 630e-9, 200e-9])  
wavelength_arr  = jnp.linspace(500e-9, 700e-9, 1000)
angle_of_incidences  = jnp.linspace(0, (70*jnp.pi/180), 1000)
polarization = 's'

R_s, T_s = tmm(material_list = material_list,
               thickness_list = thickness_list,
               wavelength_arr = wavelength_arr,
               angle_of_incidences = angle_of_incidences,
               polarization = polarization)

polarization = 'p'

R_p, T_p = tmm(material_list = material_list,
               thickness_list = thickness_list,
               wavelength_arr = wavelength_arr,
               angle_of_incidences = angle_of_incidences,
               polarization = polarization)
```

<div align="center">
  <img src="https://github.com/bahremsd/tmmax/blob/master/docs/images/thin_film_example.png" alt="usage_example">
</div>

For cases where an incoherent layer is introduced within the stack, the simulation must account for phase randomization effects. In tmmax, incoherent layers are denoted by `1`, while coherent layers remain as `0`. The following example demonstrates the configuration of the same stack with an incoherent layer:

```python
import jax.numpy as jnp
from tmmax.tmm import tmm

# Define your multilayer stack and simulation parameters

material_list = ["Air", "Y2O3", "TiO2", "Y2O3", "TiO2", "Y2O3", "TiO2", "SiO2"]
thickness_list = jnp.array([2000e-9, 100e-9, 2000e-9, 100e-9, 2000e-9, 100e-9])
coherency_list = jnp.array([1, 0, 1, 0, 1, 0])
wavelength_arr  = jnp.linspace(500e-9, 700e-9, 1000)
angle_of_incidences  = jnp.linspace(0, (70*jnp.pi/180), 1000)
polarization = 's'

R_s, T_s = tmm(material_list = material_list,
               thickness_list = thickness_list,
               wavelength_arr = wavelength_arr,
               angle_of_incidences = angle_of_incidences,
               coherency_list = coherency_list
               polarization = polarization)

polarization = 'p'

R_p, T_p = tmm(material_list = material_list,
               thickness_list = thickness_list,
               wavelength_arr = wavelength_arr,
               angle_of_incidences = angle_of_incidences,
               coherency_list = coherency_list
               polarization = polarization)
```

This approach enables precise modeling of optical interference effects in thin-film coatings, dielectric mirrors, and anti-reflective coatings, while seamlessly integrating incoherency considerations when required.

## Database

<div align="center">
  <img src="https://github.com/bahremsd/tmmax/blob/master/docs/images/SiO2_nk_plot.png" alt="database_example_sio2">
</div>

## Benchmarks

<div align="center">
  <img src="https://github.com/bahremsd/tmmax/blob/master/benchmarks/layer_size_exp_results/layer_size_figure.png" alt="layer_size_exp">
</div>

<div align="center">
  <img src="https://github.com/bahremsd/tmmax/blob/master/benchmarks/vmap_array_length_exp_results/vmap_array_length_figure.png" alt="vmap_array_length_exp">
</div>

## Installation

You can install `tmmax` via PyPI:

```bash
pip3 install tmmax
```

## License

This project is licensed under the [MIT License](https://opensource.org/license/MIT), which permits free use, modification, and distribution of the software, provided that the original copyright notice and license terms are included in all copies or substantial portions of the software. For a detailed explanation of the terms and conditions, please refer to the [LICENSE](https://github.com/bahremsd/tmmax/blob/master/LICENSE) file.

## Credits

```bibtex
@software{tmmax,
  author = {Bahrem Serhat Danis},
  title = {tmmax: transfer matrix method with jax},
  version = {1.0.0},
  url = {https://github.com/bahremsd/tmmax},
  year = {2025}
}
```

## Contact and Support

For any questions, suggestions, or issues you encounter, feel free to [open an issue](https://github.com/bahremsd/tmmax/issues) on the GitHub repository. This not only ensures that your concern is shared with the community but also allows for collaborative problem-solving and creates a helpful reference for similar challenges in the future. If you would like to collaborate or contribute to the code, you can contact me via email.

Bahrem Serhat Danis - bahremdan@gmail.com
