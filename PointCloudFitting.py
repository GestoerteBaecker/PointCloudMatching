import copy
import numpy as np
import os
from HelperFunction import *


class PointCloudFitting:
    """
    PointCloudFitting.
    Used to calculate the transformation properties between two point clouds. A 6 DOF transformation is used which describes
    the transition from the first to the second cloud.
    """
    def __init__(self, source_points: list, dest_points: list, index_table: dict) -> None:
        """
        :param source_points: first point cloud points.
        :param dest_points: second point cloud points.
        :param index_table: correspondencies between both point clouds.
        """
        self.source_points = source_points
        self.dest_points = dest_points
        self.index_table = index_table
        self.dest_vec = self.__PrepareInputData__()

        self.translation = np.zeros(3)
        self.rotation = np.eye(3)

        self.accuracys = [10**-8, 10**-5]

        self.X = np.zeros(6)
        self.x_abridged = np.ones(6)

        self.fitting_equations = self.__PrepareFittingEquation()

        self.transformation_valid = False


    def __PrepareFittingEquation(self) -> list:
        """
        Do not use this function separately.
        Assembles the fitting equations of the adjustment.
        """
        fitting_equations = []
        for id in self.index_table.keys():
            x_s, y_s, z_s = self.source_points[id]

            x_eq = lambda omega, phi, kappa, X_0, Y_0, Z_0, x=x_s, y=y_s, z=z_s: X_0 + (np.cos(kappa)*np.cos(phi)*x + (-np.sin(kappa)*np.cos(phi))*y + np.sin(phi)*z)
            y_eq = lambda omega, phi, kappa, X_0, Y_0, Z_0, x=x_s, y=y_s, z=z_s: Y_0 + ((np.sin(kappa)*np.cos(omega) + np.sin(omega)*np.sin(phi)*np.cos(kappa))*x + (-np.sin(kappa)*np.sin(omega)*np.sin(phi) + np.cos(kappa)*np.cos(omega))*y + (-np.sin(omega)*np.cos(phi))*z)
            z_eq = lambda omega, phi, kappa, X_0, Y_0, Z_0, x=x_s, y=y_s, z=z_s: Z_0 + ((np.sin(kappa)*np.sin(omega) - np.sin(phi)*np.cos(kappa)*np.cos(omega))*x + (np.sin(kappa)*np.sin(phi)*np.cos(omega) + np.sin(omega)*np.cos(kappa))*y + (np.cos(omega)*np.cos(phi))*z)

            eq = [x_eq, y_eq, z_eq]
            fitting_equations = fitting_equations + eq
        return fitting_equations


    def __PrepareInputData__(self) -> list:
        """
        Do not use this function separately.
        Converts the points from numpy arrays to the measurement vector.
        """
        dest_vec = []
        for i_source in self.index_table.keys():
            try:
                i_dest = self.index_table[i_source]
                dest_vec = np.hstack((dest_vec, self.dest_points[i_dest]))
            except:
                raise Exception("There are not corresponding points matched")
        return dest_vec


    def __MakeJacobian__(self) -> np.array:
        """
        Do not use this function separately.
        Assembles the Jacobian matrix of the adjustment.
        """
        matrix = np.zeros((len(self.fitting_equations), 6))
        for i in range(len(self.fitting_equations)):
            for j in range(6):
                matrix[i, j] = Differentiate(self.fitting_equations[i], self.X, j)
        return matrix


    def __ComputeNearing__(self, index_adding: int=0) -> np.array:
        """
        Do not use this function separately.
        Calculates a first rough estimate of the transformation parameters.
        """
        if 2+index_adding >=len(self.index_table):
            raise Exception("There are not enough (planely distributed) points given")

        ids = list(self.index_table.keys())
        p1 = self.source_points[ids[0]]
        p2 = self.source_points[ids[1]]
        p3 = self.source_points[ids[2+index_adding]]

        P1 = self.dest_points[self.index_table[ids[0]]]
        P2 = self.dest_points[self.index_table[ids[1]]]
        P3 = self.dest_points[self.index_table[ids[2 + index_adding]]]

        p12 = p2 - p1
        p13 = p3 - p1

        P12 = P2 - P1
        P13 = P3 - P1

        u = p12/np.linalg.norm(p12)
        w = np.cross(u, p13)
        if np.linalg.norm(w) < 1e-5:
            self.__ComputeNearing__(index_adding+1)
        w = w/np.linalg.norm(w)
        v = np.cross(w, u)

        U = P12/np.linalg.norm(P12)
        W = np.cross(U, P13)
        if np.linalg.norm(W) < 1e-5:
            self.__ComputeNearing__(index_adding+1)
        W = W/np.linalg.norm(W)
        V = np.cross(W, U)

        R_U_X = MakeBasis(U, V, W)
        r_u_x = MakeBasis(u, v, w)
        R = R_U_X.dot(r_u_x.T)

        phi = np.arcsin(R[0, 2])
        omega = np.arctan(-R[1, 2] / R[2, 2])
        kappa = np.arctan(-R[0, 1] / R[0, 0])

        m = float(((np.linalg.norm(P12) / np.linalg.norm(p12)) + (np.linalg.norm(P13) / np.linalg.norm(p13)) + (np.linalg.norm(P2 - P3)/np.linalg.norm(p2 - p3)))/3)
        if m < 0.5 or m > 1.5:
            raise Exception("There appears to be a transformation with scale deviating from 1")

        X_S = 1/3 * (P1 + P2 + P3)
        x_s = 1/3 * (p1 + p2 + p3)
        X_0, Y_0, Z_0 = X_S - R.dot(x_s)

        return np.array([omega, phi, kappa, float(X_0), float(Y_0), float(Z_0)])


    def __Adjust__(self) -> None:
        """
        Do not use this function separately.
        Executes the core function of the geodesic adjustement.
        """
        obs_approx = self.dest_vec * 0
        for i in range(len(self.dest_vec)):
            obs_approx[i] = self.fitting_equations[i](*self.X)
        obs_abridged = self.dest_vec - obs_approx
        A = self.__MakeJacobian__()
        n = (A.T).dot(obs_abridged)
        N = (A.T).dot(A)
        try:
            Q = np.linalg.inv(N)
        except:
            raise Exception("Adjustment not possible")
        self.x_abridged = Q.dot(n)
        self.X = self.X + self.x_abridged


    def StartAdjustment(self) -> tuple:
        """
        Calculates the transformation parameters depending on the points given and indexed in the index_table
        :return: translation vector and rotation matrix. Scale is set to 1.
        """
        if np.linalg.norm(self.X) < 1e-12:
            self.X = self.__ComputeNearing__()

        for i in range(1000):
            converged = True
            for j in range(6):
                if j < 3:
                    accuracy = self.accuracys[0]
                else:
                    accuracy = self.accuracys[1]
                converged = converged and abs(self.x_abridged[j]) < accuracy

            if not converged:
                self.__Adjust__()
            else:
                self.transformation_valid = True
                self.translation = self.X[3:6]
                self.rotation = MakeTransformationMatrix(self.X[0], self.X[1], self.X[2], 1)
                return self.translation, self.rotation
        raise Exception("Adjustment did not converge")


    def Transform(self, point: np.array) -> np.array:
        """
        Transforms a point given in the system of the first point cloud to a point in the system of the second point cloud.
        :param point: Point in system of the first point cloud
        :return: Point in system of the second point cloud
        """
        return self.translation + self.rotation.dot(point)


    def SetFittingAccuracyRotation(self, accuracy: float) -> None:
        """
        Sets the fitting acuracy of the rotation.
        :param accuracy: Accuracy for the fitting process.
        """
        self.accuracys[0] = accuracy


    def SetFittingAccuracyTranslation(self, accuracy: float) -> None:
        """
        Sets the fitting acuracy of the translation.
        :param accuracy: Accuracy for the fitting process.
        """
        self.accuracys[1] = accuracy
