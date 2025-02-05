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

## Usage

```python
from tmmax.tmm import tmm

# Define your multilayer stack and simulation parameters

material_list = ["Air", ... , "SiO2", ...]
thickness_list = jnp.array(...)
wavelength_arr  = jnp.array(...)
angle_of_incidences  = jnp.array(...)
polarization = 's' # or 'p'

result = tmm(material_list = material_list,
             thickness_list = thickness_list,
             wavelength_arr = wavelength_arr,
             angle_of_incidences = angle_of_incidences,
             polarization = polarization)
```

<div align="center">
  <img src="https://github.com/bahremsd/tmmax/blob/master/docs/images/thin_film_example.png" alt="usage_example">
</div>

```python
from tmmax.tmm import tmm

# Define your multilayer stack and simulation parameters

material_list = ["Air", ... , "SiO2", ...]
thickness_list = jnp.array(...)
wavelength_arr  = jnp.array(...)
angle_of_incidences  = jnp.array(...)
polarization = 's' # or 'p'

result = tmm(material_list = material_list,
             thickness_list = thickness_list,
             wavelength_arr = wavelength_arr,
             angle_of_incidences = angle_of_incidences,
             polarization = polarization)
```

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

## Credits

```bibtex
@software{tmmax,
  author = {Bahrem Serhat Danis},
  title = {tmmax: High-Performance Transfer Matrix Method with JAX},
  version = {0.0.2},
  url = {https://github.com/bahremsd/tmmax},
  year = {2024}
}
```

## Contact and Support

