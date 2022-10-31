import numpy as np
import copy


def Differentiate(function: callable, values: list, argument: int, h: float=0.001) -> float:
    """
    Calculates the partial derivative.
    :param function: mathematical function to be differentiated
    :param values: input arguments to the function
    :param argument: index of the argument to be differentiated
    :param h: infinitesimal for numeric derivative
    :return: derivative value
    """
    lower_values = copy.deepcopy(values)
    upper_values = copy.deepcopy(values)
    lower_values[argument] = lower_values[argument]-h
    upper_values[argument] = upper_values[argument]+h
    return (function(*upper_values) - function(*lower_values))/(2*h)


def MakeTransformationMatrix(omega: float, phi: float, kappa: float, m: float) -> np.array:
    """
    Calculates the rotation matrix
    :param omega: rotation angle x axis
    :param phi: rotation angle y axis
    :param kappa: rotation angle z axis
    :param m: scale
    :return: rotation matrix
    """
    return np.array([[m * np.cos(kappa) * np.cos(phi), m * (-np.sin(kappa) * np.cos(phi)), m * np.sin(phi)],
                     [m * (np.sin(kappa) * np.cos(omega) + np.sin(omega) * np.sin(phi) * np.cos(kappa)),
                   m * (-np.sin(kappa) * np.sin(omega) * np.sin(phi) + np.cos(kappa) * np.cos(omega)),
                   m * (-np.sin(omega) * np.cos(phi))],
                     [m * (np.sin(kappa) * np.sin(omega) - np.sin(phi) * np.cos(kappa) * np.cos(omega)),
                   m * (np.sin(kappa) * np.sin(phi) * np.cos(omega) + np.sin(omega) * np.cos(kappa)),
                   m * (np.cos(omega) * np.cos(phi))]])


def MakeBasis(a: np.array, b: np.array, c: np.array) -> np.array:
    """
    Collects the vectors as column vectors and makes a matrix
    :param a: left most column vector
    :param b: middle column vector
    :param c: right most column vector
    :return: matrix
    """
    return np.vstack((np.vstack((a, b)), c)).T


def FindKeyFromValueOfDict(map: dict, value: int) -> int:
    """
    Gets the key from the value of a dictionary
    :param map: dictionary
    :param value: value
    :return: key
    """
    for key in map:
        value_in_dict = map[key]
        if value_in_dict == value:
            return key
    return None