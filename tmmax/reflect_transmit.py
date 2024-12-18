import jax.numpy as jnp
from jax import vmap
import jax
import numpy as np
import pickle
import sys
from jax import Array
from jax.typing import ArrayLike

from .fresnel import fresnel_s, fresnel_p

#@jit
def calculate_reflectance_from_coeff(r: ArrayLike) -> Array:
    return jnp.square(jnp.abs(r))


def calculate_transmittace_from_coeff_s_pol(t: ArrayLike,
                                      nk_first_layer_of_slab: ArrayLike,
                                      angle_first_layer_of_slab: ArrayLike,
                                      nk_last_layer_of_slab: ArrayLike,
                                      angle_last_layer_of_slab: ArrayLike) -> Array:

    T = jnp.multiply(jnp.square(jnp.abs(t)), jnp.true_divide(jnp.real(jnp.multiply(nk_last_layer_of_slab, jnp.cos(angle_last_layer_of_slab))),
                                                             jnp.real(jnp.multiply(nk_first_layer_of_slab, jnp.cos(angle_first_layer_of_slab)))))
    return T

def calculate_transmittace_from_coeff_p_pol(t: ArrayLike,
                                      nk_first_layer_of_slab: ArrayLike,
                                      angle_first_layer_of_slab: ArrayLike,
                                      nk_last_layer_of_slab: ArrayLike,
                                      angle_last_layer_of_slab: ArrayLike) -> Array:

    T = jnp.multiply(jnp.square(jnp.abs(t)), jnp.true_divide(jnp.real(jnp.multiply(nk_last_layer_of_slab, jnp.conj(jnp.cos(angle_last_layer_of_slab)))),
                                                             jnp.real(jnp.multiply(nk_first_layer_of_slab, jnp.conj(jnp.cos(angle_first_layer_of_slab))))))
    return T


#@jit
def calculate_transmittace_from_coeff(t: ArrayLike,
                                      nk_first_layer_of_slab: ArrayLike,
                                      angle_first_layer_of_slab: ArrayLike,
                                      nk_last_layer_of_slab: ArrayLike,
                                      angle_last_layer_of_slab: ArrayLike,
                                      polarization: ArrayLike) -> Array:

    return jnp.select(condlist=[jnp.array_equal(polarization, jnp.array([0], dtype = jnp.int16)),
                                jnp.array_equal(polarization, jnp.array([1], dtype = jnp.int16))],
                    choicelist=[calculate_transmittace_from_coeff_s_pol(t,
                                                                        nk_first_layer_of_slab,
                                                                        angle_first_layer_of_slab,
                                                                        nk_last_layer_of_slab,
                                                                        angle_last_layer_of_slab),
                                calculate_transmittace_from_coeff_p_pol(t,
                                                                        nk_first_layer_of_slab,
                                                                        angle_first_layer_of_slab,
                                                                        nk_last_layer_of_slab,
                                                                        angle_last_layer_of_slab)])


def compute_rt_at_interface_s(layer_idx: ArrayLike,
                              nk_angles_stack: ArrayLike) -> Array:
    rt = fresnel_s(first_layer_n = nk_angles_stack.at[layer_idx,0].get(),
                   second_layer_n = nk_angles_stack.at[jnp.add(layer_idx, jnp.array([1], dtype = jnp.int32)), 0].get(),
                   first_layer_theta = nk_angles_stack.at[layer_idx, 1].get(),
                   second_layer_theta = nk_angles_stack.at[jnp.add(layer_idx, jnp.array([1], dtype = jnp.int32)), 1].get())
    #print("rt shape:  ", jnp.shape(nk_angles_stack))
    return rt

def compute_rt_at_interface_p(layer_idx: ArrayLike,
                              nk_angles_stack: ArrayLike) -> Array:
    rt = fresnel_p(first_layer_n = nk_angles_stack.at[layer_idx,0].get(),
                   second_layer_n = nk_angles_stack.at[jnp.add(layer_idx, jnp.array([1], dtype = jnp.int32)), 0].get(),
                   first_layer_theta = nk_angles_stack.at[layer_idx, 1].get(),
                   second_layer_theta = nk_angles_stack.at[jnp.add(layer_idx, jnp.array([1], dtype = jnp.int32)), 1].get())
    return rt


def vectorized_rt_s_pol():
    return vmap(compute_rt_at_interface_s, (0, None))

def vectorized_rt_p_pol():
    return vmap(compute_rt_at_interface_p, (0, None))

def polarization_based_rt_selection(layer_indices: ArrayLike, nk_angles_stack: ArrayLike, polarization: ArrayLike) -> Array:

    return jnp.select(condlist=[jnp.array_equal(polarization, jnp.array([0], dtype = jnp.int16)),
                                jnp.array_equal(polarization, jnp.array([1], dtype = jnp.int16))],
                    choicelist=[vectorized_rt_s_pol()(layer_indices, nk_angles_stack),
                                vectorized_rt_p_pol()(layer_indices, nk_angles_stack)])

#@jit
def compute_rt(nk_list: ArrayLike, angles: ArrayLike, polarization: ArrayLike) -> Array:
    """
    Calculates the angles of incidence across layers for a set of refractive indices (nk_list_2d)
    and an initial angle of incidence (initial_theta) using vectorization.

    Returns:
        jnp.ndarray: A 3D JAX array where the [i, j, :] entry represents the angles of incidence
                     for the j-th initial angle at the i-th wavelength. The size of the third dimension
                     corresponds to the number of layers.
    """
    #print(jnp.shape(nk_list))
    #print(jnp.shape(angles))
    nk_angles_stack = jnp.concat([jnp.expand_dims(nk_list, 1), jnp.expand_dims(angles, 1)], axis=1)
    #print(jnp.shape(nk_angles_stack))
    stop_value = int(jnp.size(nk_list)) - 1  # Concrete integer
    layer_indices = jnp.arange(stop_value, dtype=jnp.int32)

    return polarization_based_rt_selection(layer_indices, nk_angles_stack, polarization)
