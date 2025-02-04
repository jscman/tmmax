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

## Documentation

## Usage

<div align="center">
  <img src="https://github.com/bahremsd/tmmax/blob/master/docs/images/thin_film_example.png" alt="usage_example">
</div>

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

## License

## Credits

## Contact and Support














`tmmax` is a highly optimized Python library for simulating the optical properties of multilayer thin films using the Transfer Matrix Method (TMM). This library leverages the power of **JAX**, a modern numerical computing library, to deliver high-performance computations, automatic differentiation, and seamless GPU/TPU acceleration. By exploiting JAX’s vectorized operations and just-in-time (JIT) compilation, `tmmax` achieves unprecedented efficiency and speed in modeling complex multilayer systems.


## **Advantages**


- **Speed**: JAX's JIT compilation accelerates TMM calculations, enabling large-scale simulations to run in a fraction of the time compared to traditional methods.
- **Vectorization**: `tmmax` supports vectorized calculations for reflectance and transmittance over multiple wavelengths and angles of incidence, enhancing efficiency during parameter sweeps.
- **Automatic Differentiation**: With JAX's automatic differentiation, `tmmax` can compute gradients efficiently, facilitating the optimization of thin film designs.
- **GPU/TPU Support**: `tmmax` can offload computations to GPUs or TPUs, making it ideal for resource-intensive optical simulations.


## **Benchmarks**


![image](benchmarks/layer_size_experiment/execution_time_vs_layers.png)


The benchmark comparison demonstrates a significant computational efficiency gain when using the `tmmax` package over the widely used `tmm` library, especially for simulations involving a large number of layers. As the number of layers increases, the execution time for `tmm` scales linearly, demonstrating a performance bottleneck, while `tmmax` exhibits near-constant execution time beyond a small initial rise, suggesting superior handling of computational complexity. (Despite the increasing number of layers, both the wavelength array and theta array lengths were kept constant at 20, ensuring the results purely reflect the scaling behavior with respect to the number of layers.)


![image](benchmarks/wavelength_arr_experiment/execution_time_vs_wavelength_array_length.png)


![image](benchmarks/angle_arr_experiment/tmm_execution_time_vs_angle_length.png)


## **Installation**

You can install `tmmax` via PyPI:

```bash
pip3 install tmmax
```

## **Usage**

Here is a basic example of how to use `tmmax`:

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

For detailed usage instructions, refer to the documentation and examples provided in the repository.


## How to Cite This Repository

If you use or reference any of the templates provided in this repository, please cite it as follows:


```bibtex
@software{tmmax,
  author = {Bahrem Serhat Danis},
  title = {tmmax: High-Performance Transfer Matrix Method with JAX},
  version = {0.0.2},
  url = {https://github.com/bahremsd/tmmax},
  year = {2024}
}
```
