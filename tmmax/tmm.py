import jax
import jax.numpy as jnp # Import JAX's version of NumPy for differentiable computations
from jax import jit, vmap 

from .angle import compute_layer_angles
from .wavevector import compute_kz
from .cascaded_matmul import _cascaded_matrix_multiplication
from .data import interpolate_nk
from .reflect_transmit import _compute_rt_one_wl, _create_phases_ts_rs, _calculate_transmittace_from_coeff


@jit
def tmm_single_wl_angle_point_jit(nk_list: ArrayLike,thickness_list: ArrayLike,
                              wavelength: ArrayLike,
                              angle_of_incidence: ArrayLike,
                              polarization: ArrayLike) -> Array:


    layer_angles = compute_layer_angles(angle_of_incidence, nk_list, polarization)
    # Compute the angles within each layer based on the refractive indices, incidence angle, and wavelength
    #print("layer_angles ", layer_angles)
    #print("layer_angles shape", jnp.shape(layer_angles))
    kz = compute_kz(nk_list, layer_angles, wavelength)
    # Calculate the z-component of the wave vector for each layer
    #print("kz", kz)
    layer_phases = jnp.multiply(kz.at[1:-1].get(), thickness_list)
    # Compute the phase shifts in each layer by multiplying kz by the layer thicknesses
    # `jnp.pad(thickness_list, (1), constant_values=0)` adds a leading zero to the thickness_list
    #print("layer_phases", layer_phases)
    rt = jnp.squeeze(compute_rt(nk_list = nk_list, angles = layer_angles, polarization = polarization))
    # Compute the reflection and transmission matrices for the wavelength
    #print("rt", rt)
    #print("rt shape", jnp.shape(rt))
    tr_matrix = cascaded_matrix_multiplication(phases = layer_phases, rts = rt.at[1:,:].get())
    # Perform matrix multiplication to obtain the cascaded transfer matrix for the entire stack
    #print("tr_matrix", tr_matrix)
    tr_matrix = jnp.multiply(jnp.true_divide(1, rt.at[0,1].get()), jnp.dot(jnp.array([[1, rt.at[0,0].get()], [rt.at[0,0].get(), 1]]), tr_matrix))
    #print("tr_matrix", tr_matrix)
    # Normalize the transfer matrix and include the boundary conditions
    # `jnp.dot` multiplies the transfer matrix by the boundary conditions matrix

    r = jnp.true_divide(tr_matrix.at[1,0].get(), tr_matrix.at[0,0].get())
    t = jnp.true_divide(1, tr_matrix.at[0,0].get())
    #print("r", r)
    #print("t", t)
    # Calculate the reflectance (r) and transmittance (t) from the transfer matrix
    # Reflectance is obtained by dividing the (1, 0) element by the (0, 0) element
    # Transmittance is obtained by taking the reciprocal of the (0, 0) element

    R = calculate_reflectance_from_coeff(r)
    T = calculate_transmittace_from_coeff(t, nk_list.at[0].get(), angle_of_incidence, nk_list.at[-1].get(), layer_angles.at[-1].get(), polarization)

    #print("R", R)
    #print("T", T)
    # Compute the reflectance (R) and transmittance (T) using their respective formulas
    # Reflectance R is the squared magnitude of r
    # Transmittance T is calculated using a function `_calculate_transmittace_from_coeff`
    return R, T
    # Return the reflectance and transmittance values





def tmm_single_wl_angle_point(nk_functions: Dict[int, Callable], material_list: ArrayLike,
                               thickness_list: ArrayLike, wavelength: ArrayLike,
                               angle_of_incidence: ArrayLike, polarization: ArrayLike) -> Array:
    """
    Computes the reflectance (R) and transmittance (T) of a multi-layer optical film for a given wavelength
    and angle of incidence using the Transfer Matrix Method (TMM).

    Args:
        nk_functions (Dict[int, Callable]): Dictionary mapping material indices to functions that return
                                           the complex refractive index (n + ik) for a given wavelength.
        material_list (list[int]): List of indices representing the order of materials in the stack.
        thickness_list (jnp.ndarray): Array of thicknesses for each layer in the stack.
        wavelength (Union[float, jnp.ndarray]): Wavelength(s) of light in the simulation.
        angle_of_incidence (Union[float, jnp.ndarray]): Angle of incidence in radians.
        polarization (bool): True for TM polarization, False for TE polarization.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Reflectance (R) and transmittance (T) of the optical stack.
    """

    def get_nk_values(wl):
        """
        Retrieves the complex refractive index values for each material at the given wavelength.

        Args:
            wl (Union[float, jnp.ndarray]): Wavelength or array of wavelengths.

        Returns:
            jnp.ndarray: Array of complex refractive index values for each material.
        """
        return jnp.array([nk_functions[mat_idx](wl) for mat_idx in material_list])  # Get nk values for each material

    nk_list = get_nk_values(wavelength)  # Call get_nk_values to get refractive index values for all materials
    #print("nk_list", nk_list)
    R, T = tmm_single_wl_angle_point_jit(nk_list, thickness_list, wavelength, angle_of_incidence, polarization)
    return R, T
    # Return the reflectance and transmittance values

def tmm(material_list: List[str],
        thickness_list: jnp.ndarray,
        wavelength_arr: jnp.ndarray,
        angle_of_incidences: jnp.ndarray,
        polarization: Text) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Perform the Transfer Matrix Method (TMM) for multilayer thin films.

    Args:
        material_list (List[str]): A list of material names. Each material is identified by a string.
        thickness_list (jnp.ndarray): An array of thicknesses corresponding to each layer.
        wavelength_arr (jnp.ndarray): An array of wavelengths over which to perform the simulation.
        angle_of_incidences (jnp.ndarray): An array of angles of incidence.
        polarization (Text): The type of polarization ('s' for s-polarized or 'p' for p-polarized).

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing two JAX arrays. The first array represents the transmission coefficients, and the second array represents the reflection coefficients.
    """

    # Remove duplicate materials and create a unique set
    material_set = list(set(material_list))  # Create a unique list of materials
    # Create a mapping from material names to indices
    material_enum = {material: i for i, material in enumerate(material_set)}  # Map each material to an index
    # Convert the original material list to a list of indices
    material_list = [int(material_enum[material]) for material in material_list]  # Map materials to indices based on material_enum
    # Create a dictionary of interpolation functions for each material
    nk_funkcs = {i: interpolate_nk(material) for i, material in enumerate(material_set)}  # Interpolate n and k for each material

    if polarization == 's':
        # Unpolarized case: Return tuple (s-polarization, p-polarization)
        polarization = jnp.array([0], dtype = jnp.int16)
    elif polarization == 'p':
        # s-polarization case
        polarization = jnp.array([1], dtype = jnp.int16)
    else:
        raise TypeError("The polarization can be 's' or 'p', not the other parts. Correct it")  # Raise an error for invalid polarization input


    # Vectorize the _tmm_single_wl_angle_point function across wavelength and angle of incidence
    tmm_vectorized = vmap(vmap(tmm_single_wl_angle_point, (None, None, None, 0, None, None)), (None, None, None, None, 0, None))  # Vectorize _tmm_single_wl_angle_point over wavelengths and angles

    # Apply the vectorized TMM function to the arrays
    result = tmm_vectorized(nk_funkcs, material_list, thickness_list, wavelength_arr, angle_of_incidences, polarization)  # Compute the TMM results

    # Return the computed result
    return result  # Tuple of transmission and reflection coefficients