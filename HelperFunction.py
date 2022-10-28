import numpy as np


def Differentiate(function, values, argument, h=0.001):
    lower_values = copy.deepcopy(values)
    upper_values = copy.deepcopy(values)
    lower_values[argument] = lower_values[argument]-h
    upper_values[argument] = upper_values[argument]+h
    return (function(*upper_values) - function(*lower_values))/(2*h)


def MakeTransformationMatrix(omega, phi, kappa, m):
    return np.array([[m * np.cos(kappa) * np.cos(phi), m * (-np.sin(kappa) * np.cos(phi)), m * np.sin(phi)],
                     [m * (np.sin(kappa) * np.cos(omega) + np.sin(omega) * np.sin(phi) * np.cos(kappa)),
                   m * (-np.sin(kappa) * np.sin(omega) * np.sin(phi) + np.cos(kappa) * np.cos(omega)),
                   m * (-np.sin(omega) * np.cos(phi))],
                     [m * (np.sin(kappa) * np.sin(omega) - np.sin(phi) * np.cos(kappa) * np.cos(omega)),
                   m * (np.sin(kappa) * np.sin(phi) * np.cos(omega) + np.sin(omega) * np.cos(kappa)),
                   m * (np.cos(omega) * np.cos(phi))]])

def MakeBasis(a, b, c):
    return np.vstack((np.vstack((a, b)), c)).T


def FindKeyFromValueOfDict(map: dict, value: int) -> int:
    for key in map:
        value_in_dict = map[key]
        if value_in_dict == value:
            return key
    return None