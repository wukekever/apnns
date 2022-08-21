import json
from math import pi

import numpy as np

# ===================================================================
# Utility functions for reading config files
# ===================================================================


class DictionaryUtility:
    """
    Utility methods for dealing with dictionaries.
    """

    @staticmethod
    def to_object(item):
        """
        Convert a dictionary to an object (recursive).
        """

        def convert(item):
            if isinstance(item, dict):
                return type("jo", (), {k: convert(v) for k, v in item.items()})
            if isinstance(item, list):

                def yield_convert(item):
                    for index, value in enumerate(item):
                        yield convert(value)

                return list(yield_convert(item))
            else:
                return item

        return convert(item)

    def to_dict(obj):
        """
        Convert an object to a dictionary (recursive).
        """

        def convert(obj):
            if not hasattr(obj, "__dict__"):
                return obj
            result = {}
            for key, val in obj.__dict__.items():
                if key.startswith("_"):
                    continue
                element = []
                if isinstance(val, list):
                    for item in val:
                        element.append(convert(item))
                else:
                    element = convert(val)
                result[key] = element
            return result

        return convert(obj)


def get_config(config_path):
    with open(config_path) as json_data_file:
        config = json.load(json_data_file)
    # convert dict to object recursively for easy call
    config = DictionaryUtility.to_object(config)
    return config


# =========================================================
# Utility functions for bgk_solver
# =========================================================

EPS = 1e-8


def maxwellian(v, rho, u, T):
    return rho / np.sqrt(2 * pi * T) * np.exp(-((v - u) ** 2) / (2 * T))


def Maxwellian(v, rho, u, T):
    return (rho / np.sqrt(2 * pi * T))[:, None] * np.exp(
        -((v - u[:, None]) ** 2) / (2 * T[:, None])
    )


# 1st order upwind flux
def F_p(psi, dx):
    return (psi[1:-1] - psi[:-2]) / dx


def F_m(psi, dx):
    return (psi[2:] - psi[1:-1]) / dx


# 2nd order flux using BAP average as limiter
def B(x):
    return x / np.sqrt(1 + x**2)


def inv_B(x):
    return x / np.sqrt(1 - x**2)


def BAP(s_l, s_r):
    return inv_B(0.5 * (B(s_l) + B(s_r)))


def slope(F, dx):
    return BAP((F[1:-1] - F[:-2]) / dx, (F[2:] - F[1:-1]) / dx)


def F_p_BAP(F, v, dx, dt):
    s = slope(F, dx)
    F = F.copy()
    F[1:-1] += 0.5 * dx * s
    return (F[2:-2] - F[1:-3]) / dx


def F_m_BAP(F, v, dx, dt):
    s = slope(F, dx)
    F = F.copy()
    F[1:-1] -= 0.5 * dx * s
    return (F[3:-1] - F[2:-2]) / dx


# 2nd order flux using combination of F_L (upwind) and F_H (Lax-Wendroff) and van_leer_limiter (can be changed to others).
def van_leer_limiter(r):
    return (r + np.abs(r)) / (1.0 + np.abs(r))


# flux for upwind direction
def F_p_2(f, vp, dx, dt, limiter=van_leer_limiter):
    r = (f[1:-1] - f[:-2]) / (f[2:] - f[1:-1] + EPS)
    phi = limiter(r)
    F = f.copy()
    F[1:-1] += 0.5 * phi * (1.0 - vp * dt / dx) * (f[2:] - f[1:-1])
    return (F[2:-2] - F[1:-3]) / dx


# flux for downwind direction
def F_m_2(f, vm, dx, dt, limiter=van_leer_limiter):
    r = (f[2:] - f[1:-1]) / (f[1:-1] - f[:-2] + EPS)
    phi = limiter(r)
    F = f.copy()
    F[1:-1] += 0.5 * phi * (-1.0 - vm * dt / dx) * (f[1:-1] - f[:-2])
    return (F[3:-1] - F[2:-2]) / dx
