import jax.numpy as jnp
from jax import vmap
import jax
import numpy as np
import pickle
import sys
from jax import Array
from jax.typing import ArrayLike

#@jit
def compute_kz(nk_list: ArrayLike,
               angles: ArrayLike,
               wavelength: ArrayLike) -> Array:

    kz = jnp.true_divide(jnp.multiply(jnp.multiply(jnp.array([2.0]), jnp.pi), jnp.multiply(nk_list, jnp.cos(angles))), wavelength)

    return kz