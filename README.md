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

`tmmax` is a high-performance computational library designed for efficient calculation of optical properties in multilayer thin-film structures. Engineered to serve as a Swiss Army knife tool for thin-film optics research, `tmmax` integrates a comprehensive suite of advanced numerical methods. At its core, `tmmax` leverages JAX to enable just-in-time (JIT) compilation, vectorization and XLA (Accelerated Linear Algebra) operations, dramatically accelerating the evaluation of optical responses in multilayer coatings. By exploiting these capabilities, `tmmax` achieves exceptional computational speed, making it the optimal choice for modeling and analyzing complex systems.

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

The `tmmax` repository features a database of 29 materials commonly used in optical thin film research. These materials is selected based on their prevalence in the field, as highlighted in the 18th chapter of _Thin-Film Optical Filters, Fifth Edition by H. Angus Macleod_ [doi](https://doi.org/10.1201/b21960). However, it’s important to clarify that these materials were not directly sourced from the book; rather, the book provides a reference for materials typically utilized in thin-film optics studies. The material data, including refractive index (n) and extinction coefficient (k) values, is available in the `tmmax/nk_data` folder in both .csv and .npy formats. While the initial data was stored in .csv format, the repository switched to .npy to leverage JAX’s faster data loading capabilities, as reading .csv files was slower in comparison.

Most of the refractive index and extinction coefficient data for the materials in the database has been obtained from [refractiveindex.info](https://refractiveindex.info/), which itself aggregates data from various research articles. To ensure proper attribution, we will provide references to the original sources for each material in the `docs/database_info` folder. You can access these references by reviewing the `README` file in that directory. 

For example, to visualize the n and k data for SiO2, a material with widely accepted optical properties, you can use the `visualize_material_properties(material_name = "SiO2")` function in the `tmmax/data`. This allows for a straightforward representation of the material's refractive index and extinction coefficient.


<div align="center">
  <img src="https://github.com/bahremsd/tmmax/blob/master/docs/images/SiO2_nk_plot.png" alt="database_example_sio2">
</div>

The database is designed to be extensible, and we plan to include additional materials in future versions of tmmax. Contributions are welcome, and if you have a material you would like to add, please feel free to open an issue or submit a pull request. 

## Benchmarks

Farklı transfer matrix metodlarının benchmark'larını yaparken ilk önce layer size'a göre run time'ın comparison2unu yaptık. Çünkü optical multilayer coating'ler hassasiyetine göre çok fazla layer'lı olabilirler. Bu yüzden tmm, tmm-fast, vtmm ve tmmax'ı aynı koşullar altında (aynı hardware ve aynı wavelength, anlge of incidence array length'i) comparison'unu yaptık. Gördüğümüz kadarıyla layer sayısı arttığında en scalable sistem bizimkisi yani tmmax oluyor.


Wavelength array length'ini ve angle of incidence array length'leri de run time'a etkisi vardır. Bundan dolayı bu iki aprametreye göre de run time analziei yapıldı. Bu analizde fast-tmm'in vtmm ve tmmax'a göre daha yavaş kaldığı görüldü. Ancak fast-tmm CPU'da multi core processing'e göre tasarlandığı ayrıca GPU için de tasarlandığı için onun bu preformans değerleri gayet uyugndur. Denildiği gibi fast-tmm 1 core cpu için tasarlanmamıştır. Bunların yanında vtmm ile tmmax'ın array length'lerine göre performansı neredeyse aynı olsa da layer sayısı 8'den 80'e arttığında vtmm'in tmmax'dan daha fazla zamn aldığını görebiliyoruz. Ayrıca benchmark'ın yapıjldığı zaman vtmm'in inital beta sürümüdür bundan dolayı ilerleyen vtmm sürümlerinde bu benchmark tekrar yapılmalıdır. 

Yukarıda da bahsettiğimiz gibi bazı tmm approach'leri gpu üzerinde daha iyi çalışacaktır çünkü vtmm tensorflow, fast-tmm pytoch framework2ünde yazılmıştır. Eğer elinizde GPU bulunuyorsa bu benchmark'ları baz almayarak tekrar vtmm, fast-tmm ve tmmax'ın benchmark'larını yapınız. Benchmark sonuçları ve kodları benchmark klasöründe bulunabilir.

In evaluating the performance of various transfer matrix method implementations, we conducted rigorous benchmarking to compare runtime efficiency as a function of layer count, wavelength array length, and angle of incidence array length. Given that optical multilayer coatings can contain a substantial number of layers based on design constraints and performance sensitivity, scalability is a critical factor. To ensure a fair comparison, all tests were executed under identical conditions, including hardware specifications and input parameters.

### Layer Size vs Run Time

One of the primary factors influencing computational complexity in tmm simulations is the number of layers in the multilayer stack. We benchmarked [`tmm`](https://github.com/sbyrnes321/tmm), [`tmm-fast`](https://github.com/MLResearchAtOSRAM/tmm_fast), [`vtmm`](https://github.com/fancompute/vtmm), and `tmmax` to assess their performance under increasing layer counts. The results indicate that as the number of layers grows, tmmax demonstrates good scalability compared to other implementations. This makes tmmax particularly well-suited for simulating highly complex multilayer structures without significant degradation in performance.

<div align="center">
  <img src="https://github.com/bahremsd/tmmax/blob/master/benchmarks/layer_size_exp_results/layer_size_figure.png" alt="layer_size_exp">
</div>

### Wavelength and Angle of Incidence Array Lnegths vs Run Time

Apart from layer count, the length of the wavelength array and angle of incidence array significantly impact computational performance. Our analysis revealed that `tmm-fast` exhibited slower execution times relative to `vtmm` and `tmmax` under single-threaded execution. However, it is important to note that `tmm-fast` is explicitly optimized for multi-core CPU execution and GPU acceleration. Thus, benchmarking `tmm-fast` on a single-core CPU does not reflect its full performance potential.

When comparing `vtmm` and `tmmax`, both demonstrated similar runtime performance across varying array lengths. However, as the layer count increased from 8 to 80, `vtmm` exhibited a higher runtime overhead than `tmmax`, reinforcing the latter’s efficiency in handling deep multilayer stacks. Additionally, it is worth noting that these benchmarks were conducted on an "Initial beta" version of `vtmm`, and future iterations may yield different performance characteristics. Therefore, a reevaluation on updated `vtmm` releases is recommended.

<div align="center">
  <img src="https://github.com/bahremsd/tmmax/blob/master/benchmarks/vmap_array_length_exp_results/vmap_array_length_figure.png" alt="vmap_array_length_exp">
</div>

As discussed, certain tmm implementations leverage GPU acceleration for enhanced computational efficiency. Specifically, `vtmm` is implemented in TensorFlow, while `tmm-fast` is developed using Pytorch with multi-threading and GPU compatibility. Consequently, if a GPU is available, the benchmarks presented here may not be representative, and users are encouraged to perform their own comparative analyses of `vtmm`, `fast-tmm`, and `tmmax` under GPU-accelerated conditions. For a detailed breakdown of benchmark results and implementation details, refer to the `benchmark` directory within the repository.

## Installation

You can install `tmmax` via PyPI:

```bash
pip3 install tmmax
```

## License

This project is licensed under the [MIT License](https://opensource.org/license/MIT), which permits free use, modification, and distribution of the software, provided that the original copyright notice and license terms are included in all copies or substantial portions of the software. For a detailed explanation of the terms and conditions, please refer to the [LICENSE](https://github.com/bahremsd/tmmax/blob/master/LICENSE) file.

## Credits

This repository sometimes utilizes approaches from Steven Byrnes' [`tmm`](https://github.com/sbyrnes321/tmm) library, and we thank him for its development. In the context of calculating the optical properties of incoherent multilayer thin films, we used the approach from a paper by _Charalambos C. Katsidis and Dimitrios I. Siapkas, "General transfer-matrix method for optical multilayer systems with coherent, partially coherent, and incoherent interference"_ (Applied Optics, 41(19):3978) 2002 [doi](https://doi.org/10.1364/AO.41.003978).

For further exploration, we also recommend examining the [`vtmm`](https://github.com/fancompute/vtmm) and [`tmm-fast`](https://github.com/MLResearchAtOSRAM/tmm_fast) libraries, which offer alternative approaches for thin film simulation. In particular, the tmm-fast library integrates GPU acceleration via the PyTorch framework, providing a significant performance boost for batched simulations. We strongly encourage leveraging this capability if you have access to GPU resource.

Also if you find the `tmmax` library beneficial in your work, we kindly ask that you consider citing us.

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
