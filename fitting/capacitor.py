import numpy as np
from scipy.special import ellipk


EPS0 = 8.85418782 # pf/m
LOG_PI_CORR = np.log(16) / np.pi


def elliptic_modulus(gap, film_thickness, unit_cell):
    """
    Calculate m (k^2) for interdigital using Wu model
    All the inputs need to be the same units.
    """
    pi_over_4h = 0.25 * np.pi / film_thickness

    s = np.sinh(pi_over_4h * (3 * unit_cell - gap))
    s_ratio1 = s / np.sinh(pi_over_4h * (unit_cell + gap))
    s_ratio2 = s / np.sinh(pi_over_4h * (unit_cell - gap))

    return (s_ratio1 - 1) * (s_ratio1 + 1) / ((s_ratio2 - 1) * (s_ratio2 + 1))


def geometry_thin_film(gap: float, film_thickness: float, unit_cell: float=20,
                       finger_length:float = 1e-3, finger_num: int=50):
    """
    Calculate the geometric factor for a thin film on an interdigital capacitor 
    Delta C* = eps0 * (eps - 1) * geometric_factor
    :param gap: gap width in micron
    :param film_thickness: thickness of film in micron
    :param unit_cell: width of a gap and a finger combined in micron (always 20)
    :param finger_length: length of strips of the capacitor in meters (always 1 mm or 1e-3)
    :param finger_num: number of fingers of the capacitor.
    :return: Geometric factor in meters
    """
    elliptic_modulus_m = elliptic_modulus(gap, film_thickness, unit_cell)

    elliptic_integral_ratio = np.pi / np.log(16. / elliptic_modulus_m)

    return (finger_num - 1) * finger_length * elliptic_integral_ratio


def geometry_thick_film(gap: float, film_thickness: float, unit_cell: float=20,
                        finger_length:float = 1e-3, finger_num: int=50):
    """
    Calculate the geometric factor for a thick film on an interdigital capacitor 
    Delta C* = eps0 * (eps - 1) * geometric_factor
    :param gap: gap width in micron
    :param film_thickness: thickness of film in micron
    :param unit_cell: width of a gap and a finger combined in micron (always 20)
    :param finger_length: length of strips of the capacitor in meters (always 1 mm or 1e-3)
    :param finger_num: number of fingers of the capacitor.
    :return: Geometric factor in meters
    """
    elliptic_modulus_m = elliptic_modulus(gap, film_thickness, unit_cell)

    elliptic_integral_ratio = ellipk(elliptic_modulus_m) / ellipk(1 - elliptic_modulus_m)

    return (finger_num - 1) * finger_length * elliptic_integral_ratio


def geometry_parallel_plate(gap: float, film_thickness: float, unit_cell: float=20,
                        finger_length:float = 1e-3, finger_num: int=50):
    """
    Calculate the geometric factor for a thin film on an interdigital capacitor using parallel plate approximation
    Delta C* = eps0 * (eps - 1) * geometric_factor
    :param gap: gap width in micron
    :param film_thickness: thickness of film in micron
    :param unit_cell: width of a gap and a finger combined in micron (always 20)
    :param finger_length: length of strips of the capacitor in meters (always 1 mm or 1e-3)
    :param finger_num: number of fingers of the capacitor.
    :return: Geometric factor in meters
    """
    return (finger_num - 1) * finger_length * film_thickness / gap

def bare(gap: float, unit_cell: float=20, silica_constant: float=3.9, finger_length:float=1e-3, finger_num: int=50):
    return EPS0 * geometry_thick_film(gap, 500, unit_cell, finger_length, finger_num) * (1 + silica_constant)

def dielectric_constant(delta_cap_real: float, delta_cap_imag: float, gap: float, film_thickness: float,
                        finger_length:float = 1e-3, finger_num: int=50):
    inv_Cx = (LOG_PI_CORR + gap / film_thickness) / (EPS0 * (finger_num - 1) * finger_length)
    eps_real = delta_cap_real * inv_Cx + 1
    eps_imag = delta_cap_imag * inv_Cx
    return eps_real, eps_imag

def dielectric_constant2(delta_cap_real: float, delta_cap_imag: float, gap: float, film_thickness: float,
                         finger_length:float = 1e-3, finger_num: int=50,
                         delta_cap_real_err: float=0., delta_cap_imag_err: float=0., gap_err: float=0.,
                         film_thickness_err: float=0., finger_length_err: float=0.):
    inv_Cx = (LOG_PI_CORR + gap / film_thickness) / (EPS0 * (finger_num - 1) * finger_length)
    eps_real = delta_cap_real * inv_Cx + 1
    eps_imag = delta_cap_imag * inv_Cx

    inv_Cx_no_corr = gap / (EPS0 * (finger_num - 1) * finger_length * film_thickness)
    der_cap = inv_Cx
    der_len_real = eps_real / finger_length
    der_len_imag = eps_imag / finger_length
    der_gap_real = inv_Cx_no_corr * delta_cap_real / gap
    der_gap_imag = inv_Cx_no_corr * delta_cap_imag / gap
    der_thk_real = inv_Cx_no_corr * delta_cap_real / film_thickness
    der_thk_imag = inv_Cx_no_corr * delta_cap_imag / film_thickness
    term_cap_real = delta_cap_real_err * der_cap
    term_cap_real *= term_cap_real
    term_cap_imag = delta_cap_imag_err * der_cap
    term_cap_imag *= term_cap_imag
    term_len_real = finger_length_err * der_len_real
    term_len_real *= term_len_real
    term_len_imag = finger_length_err * der_len_imag
    term_len_imag *= term_len_imag
    term_gap_real = gap_err * der_gap_real
    term_gap_real *= term_gap_real
    term_gap_imag = gap_err * der_gap_imag
    term_gap_imag *= term_gap_imag
    term_thk_real = film_thickness_err * der_thk_real
    term_thk_real *= term_thk_real
    term_thk_imag = film_thickness_err * der_thk_imag
    term_thk_imag *= term_thk_imag

    # error
    real_err = np.sqrt(term_cap_real + term_len_real + term_gap_real + term_thk_real)
    imag_err = np.sqrt(term_cap_imag + term_len_imag + term_gap_imag + term_thk_imag)

    return (eps_real, eps_imag), (real_err, imag_err)

def susceptibility_pp(delta_cap_real: float, delta_cap_imag: float, gap: float, film_thickness: float,
                   unit_cell: float=20, finger_length:float = 1e-3, finger_num: int=50,
                   delta_cap_real_err: float=0., delta_cap_imag_err: float=0., gap_err: float=0.,
                   film_thickness_err: float=0., finger_length_err: float=0.):
    inv_Cx = gap / (EPS0 * (finger_num - 1) * finger_length * film_thickness)
    real_susceptibility = delta_cap_real * inv_Cx
    imag_susceptibility = delta_cap_imag * inv_Cx

    # derivatives of susceptibility with respect to other values
    der_cap = inv_Cx
    der_len_real = real_susceptibility / finger_length
    der_len_imag = imag_susceptibility / finger_length
    der_gap_real = real_susceptibility / gap
    der_gap_imag = imag_susceptibility / gap
    der_thk_real = real_susceptibility / film_thickness
    der_thk_imag = imag_susceptibility / film_thickness

    # squares of (errors times derivatives)
    term_cap_real = delta_cap_real_err * der_cap
    term_cap_real *= term_cap_real
    term_cap_imag = delta_cap_imag_err * der_cap
    term_cap_imag *= term_cap_imag
    term_len_real = finger_length_err * der_len_real
    term_len_real *= term_len_real
    term_len_imag = finger_length_err * der_len_imag
    term_len_imag *= term_len_imag
    term_gap_real = gap_err * der_gap_real
    term_gap_real *= term_gap_real
    term_gap_imag = gap_err * der_gap_imag
    term_gap_imag *= term_gap_imag
    term_thk_real = film_thickness_err * der_thk_real
    term_thk_real *= term_thk_real
    term_thk_imag = film_thickness_err * der_thk_imag
    term_thk_imag *= term_thk_imag

    # error
    real_err = np.sqrt(term_cap_real + term_len_real + term_gap_real + term_thk_real)
    imag_err = np.sqrt(term_cap_imag + term_len_imag + term_gap_imag + term_thk_imag)

    return (real_susceptibility, imag_susceptibility), (real_err, imag_err)



def susceptibility(delta_cap_real: float, delta_cap_imag: float, gap: float, film_thickness: float,
                   unit_cell: float=20, finger_length:float = 1e-3, finger_num: int=50,
                   delta_cap_real_err: float=0., delta_cap_imag_err: float=0., gap_err: float=0.,
                   film_thickness_err: float=0., finger_length_err: float=0.):
    common_denom = 1. / (EPS0 * (finger_num - 1) * finger_length)
    log16_over_pi = np.log(16.) / np.pi
    gap_thick_ratio = gap / film_thickness
    inv_Cx = common_denom * (log16_over_pi + gap_thick_ratio)
    real_susceptibility = delta_cap_real * inv_Cx
    imag_susceptibility = delta_cap_imag * inv_Cx
    
    # derivatives of susceptibility with respect to other variables
    der_cap = inv_Cx
    der_len_real = real_susceptibility / finger_length
    der_len_imag = imag_susceptibility / finger_length
    der_gap_real = delta_cap_real * common_denom / film_thickness
    der_gap_imag = delta_cap_imag * common_denom / film_thickness
    der_thk_real = der_gap_real * gap_thick_ratio
    der_thk_imag = der_gap_imag * gap_thick_ratio

    # squares of (errors times derivatives)
    term_cap_real = delta_cap_real_err * der_cap
    term_cap_real *= term_cap_real
    term_cap_imag = delta_cap_imag_err * der_cap
    term_cap_imag *= term_cap_imag
    term_len_real = finger_length_err * der_len_real
    term_len_real *= term_len_real
    term_len_imag = finger_length_err * der_len_imag
    term_len_imag *= term_len_imag
    term_gap_real = gap_err * der_gap_real
    term_gap_real *= term_gap_real
    term_gap_imag = gap_err * der_gap_imag
    term_gap_imag *= term_gap_imag
    term_thk_real = film_thickness_err * der_thk_real
    term_thk_real *= term_thk_real
    term_thk_imag = film_thickness_err * der_thk_imag
    term_thk_imag *= term_thk_imag

    # error
    real_err = np.sqrt(term_cap_real + term_len_real + term_gap_real + term_thk_real)
    imag_err = np.sqrt(term_cap_imag + term_len_imag + term_gap_imag + term_thk_imag)

    return (real_susceptibility, imag_susceptibility), (real_err, imag_err)


if __name__ == "__main__":
    import sys

    gap = float(sys.argv[1])
    film_thickness = float(sys.argv[2])

    g_thin = geometry_thin_film(gap, film_thickness)
    g_thick = geometry_thick_film(gap, film_thickness)
    g_pp = geometry_parallel_plate(gap, film_thickness)

    print("Using Teddy's thin film approximation")
    print(f"G = {g_thin:.8e} m")
    print(f"C = {g_thin * EPS0:.8e} fF\n")

    print("Using scipy elliptic integrals")
    print(f"G = {g_thick:.8e} m")
    print(f"C = {g_thick * EPS0:.8e} fF\n")

    print("Using parallel plate approximation")
    print(f"G = {g_pp:.8e} m")
    print(f"C = {g_pp * EPS0:.8e} fF")