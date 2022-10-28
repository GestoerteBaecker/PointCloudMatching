import numpy as np
import copy

from HelperFunction import *


class PointCloud:
    """
    PointCloud.
    Used to match smaller point clouds to each other. The inner geometry (distances) of these two clouds has to be the same
    (rigid body transformation, but with a certain threshold of change of inner geometry).
    One purpose would be a geodetic measurement of marked points.

    Use: Instantiate an object of this class by giving a point cloud. Match / assign a second point cloud by using the method
    'Assign'. The output is a dictionary which maps points of the source point cloud to points of the target / second point cloud.
    Example: {0: 2, 1:1, 2:0} The point 0 of the source point cloud is equal to the point 2 of the target point cloud.
    """

    def __init__(self, points: list, threshold: float) -> None:
        """
        :param points: list of np.arrays of the same dimension.
        :param threshold: defines the maximum deviation of two distances to be identified as equal
        """
        self.points = points
        self.threshold = threshold
        self.distances, self.max_distance = PointCloud.CalculateDistanceMatrix(points)
        self.number_of_points = len(self.points)
        self.registration = None, None
        self.points_to_match = None
        self.distances_to_match, self.max_distance_to_match = None, None
        self.index_table = dict()


    @staticmethod
    def CalculateDistanceMatrix(points: list) -> tuple:
        """
        Calculates the distance matrix of all points.
        :param points: list of np.arrays of the same dimension.
        :return: quadratic matrix of dimension len(points) and the maximum distance between two points of the cloud
        """
        number_of_points = len(points)
        distances = np.zeros((number_of_points, number_of_points))
        max_distance = 0
        for i in range(number_of_points):
            for j in range(i+1, number_of_points):
                if i == j:
                    continue
                distances[i, j] = np.linalg.norm(points[i] - points[j])
                distances[j, i] = distances[i, j]
                if distances[i, j] > max_distance:
                    max_distance = distances[i, j]
        return distances, max_distance


    def FindDistance(self, distance_matrix: np.array, distance: float) -> tuple:
        """
        Finds the given distance in the given distance matrix.
        :param distance_matrix: quadratic matrix.
        :param distance: distance which has to be find in the distance matrix.
        :return: if found, output is the indexes. If not found output is (-1,-1)
        """
        number_of_points = distance_matrix.shape[0]
        for i in range(number_of_points):
            for j in range(i+1, number_of_points):
                if abs(distance_matrix[i, j] - distance) < self.threshold:
                    return (i, j)
        return (-1, -1)


    def FindSubsetOfDistanceMatrixWithMaximumDistances(self, number_of_points_to_use) -> np.array:
        dim = self.distances.shape[0]
        if dim <= number_of_points_to_use:
            return self.distances
        distances = []
        for i in range(dim):
            for j in range(i+1,dim):
                distances.append(self.distances[i,j])
        indices = [key_val[0] for key_val in sorted(enumerate(distances), key=lambda x:x[1])]

        old_indices_already_added = []
        new_points = []
        new_distance_matrix = np.zeros((dim, dim))
        for i in range(len(distances)):
            if indices[i] < len(distances) - number_of_points_to_use:
                continue

            if indices[i] in old_indices_already_added

            if len(new_points) == number_of_points_to_use:
                return new_distance_matrix, new_points


    def Preassign(self, target_point_cloud: list, expected_blunder_ratio: float=0.0) -> dict:
        """
        Finds the points in a second point cloud which correspond to the source point cloud. These correspondencies are
        only calculated for 10 points or an expected number of blunder points (+3 for calculating the registration) due to
        performance reasons.
        :param target_point_cloud:
        :param expected_blunder_ratio: ratio of blunders to real number of points
        :return: dictionary. 'key' is the index of the points from 'self.points'; 'value' is the index of the corresponding
        points from 'target_point_cloud'.
        """
        self.points_to_match = target_point_cloud
        target_number_of_all_points = len(target_point_cloud)
        expected_blunders = target_number_of_all_points * expected_blunder_ratio
        target_number_of_points = min([max([expected_blunders+3, 10]), target_number_of_all_points])
        target_point_cloud_reduced = [copy.deepcopy(target_point_cloud[i]) for i in range(target_number_of_points)]
        self.distances_to_match, self.max_distance_to_match = PointCloud.CalculateDistanceMatrix(target_point_cloud_reduced)
        self.index_table = dict()
        altered_indexes_correspondencies_counter = dict()
        for i in range(self.number_of_points):
            correspondencies = [0] * target_number_of_points
            for j in range(self.number_of_points):
                source_distance = self.distances[i, j]
                target_indexes = self.FindDistance(self.distances_to_match, source_distance)
                if target_indexes == (-1, -1):
                    continue
                correspondencies[target_indexes[0]] += 1
                correspondencies[target_indexes[1]] += 1
            if all(el == 0 for el in correspondencies):
                continue
            if correspondencies.count(max(correspondencies)) > 1:
                continue
            # if the point with index i is not contained in the target point cloud, it could happen that other
            # distances are equal to the compared distance and hence wrong points are added to the correspondence.
            # Those points are filtered with this condition
            if max(correspondencies) >= min((self.number_of_points / 2) - 1, 3):
                altered_index = correspondencies.index(max(correspondencies))
                if altered_index in altered_indexes_correspondencies_counter.keys():
                    if altered_indexes_correspondencies_counter[altered_index] < max(correspondencies):
                        index_to_delete = FindKeyFromValueOfDict(self.index_table, altered_index)
                        del self.index_table[index_to_delete]
                        self.index_table[i] = altered_index
                        altered_indexes_correspondencies_counter[altered_index] = max(correspondencies)
                else:
                    self.index_table[i] = altered_index
                    altered_indexes_correspondencies_counter[altered_index] = max(correspondencies)
        return self.index_table


    def __FindIndicesMaxDistanceOfCorrespondencies__(self, distance_matrix: np.array, index_table: dict) -> tuple:
        """
        Do not use this function separately.
        Finds the indices of the maximum distance in the distance matrix.
        """
        dim = distance_matrix.shape[0]
        max_distance = 0
        indices = None, None
        correspondencies = index_table.values()
        for i in range(dim):
            for j in range(i+1, dim):
                if distance_matrix[i, j] > max_distance and i in correspondencies and j in correspondencies:
                    max_distance = distance_matrix[i, j]
                    indices = i, j
        return indices


    def Assign(self, target_point_cloud: list = list(), expected_blunder_ratio: float=0.0) -> dict:
        """
        Finds the points in a second point cloud which correspond to the source point cloud.
        :param target_point_cloud:
        :param expected_blunder_ratio: ratio of blunders to real number of points
        :return: dictionary. 'key' is the index of the points from 'self.points'; 'value' is the index of the corresponding
        points from 'target_point_cloud'.
        """
        if len(target_point_cloud) != 0:
            self.Preassign(target_point_cloud, expected_blunder_ratio)
            if len(self.index_table) == len(target_point_cloud):
                return self.index_table

        index_1, index_2 = self.__FindIndicesMaxDistanceOfCorrespondencies__(self.distances_to_match, self.index_table) #self.FindDistance(self.distances_to_match, self.max_distance_to_match)
        #print(self.index_table)
        source_points = [self.points[FindKeyFromValueOfDict(self.index_table, index_1)], self.points[FindKeyFromValueOfDict(self.index_table, index_2)], None]
        destination_points = [self.points_to_match[index_1], self.points_to_match[index_2], None]

        for third_index in self.index_table.values():
            if third_index == index_1 or third_index == index_2:
                continue
            source_points[2] = self.points[FindKeyFromValueOfDict(self.index_table, third_index)]
            destination_points[2] = self.points_to_match[third_index]
            try:
                transformation = self.Register(source_points, destination_points)
            except:
                continue
            if transformation is not None:
                self.registration = transformation
                self.__AssignByCoordinates__()
                return self.index_table

        raise Exception("There are not enough points or the geometry of the distribution is invalid")

    def Transform(self, point: np.array) -> np.array:
        """
        Transforms a point from the destination point cloud system to the system of the source point cloud.
        :param point: Point in system of the destination point cloud.
        :return: Point in system of the source point cloud.
        """
        if self.registration[0] is None:
            return
        return self.registration[0] + self.registration[1].dot(point)

    def __AssignByCoordinates__(self) -> None:
        """
        Do not use this function separately.
        Matches points based on the their position in space.
        """
        if self.registration[0] is None:
            raise Exception("Please call 'Assign' instead")
        for i_second, point in enumerate(self.points_to_match):
            if i_second in self.index_table.values():
                continue
            point_second = self.Transform(point)
            for i_first, point_first in enumerate(self.points):
                if i_first in self.index_table.keys():
                    continue
                x_equals = abs(point_first[0] - point_second[0]) < self.threshold
                y_equals = abs(point_first[1] - point_second[1]) < self.threshold
                z_equals = abs(point_first[2] - point_second[2]) < self.threshold
                if x_equals and y_equals and z_equals:
                    self.index_table[i_first] = i_second
                    break

    def __Preregister__(self, source_points: list, destination_points: list) -> tuple:
        """
        Calculates the transformation between the two given point lists.
        :param source_points: First point cloud
        :param destination_points: Second point cloud
        :return: Transformation (translation and rotation).
        """
        if (len(source_points) != len(destination_points)) or (len(source_points) != 3):
            raise Exception("There are not enough points")
        p1, p2, p3 = source_points
        P1, P2, P3 = destination_points

        p12 = p2 - p1
        p13 = p3 - p1

        P12 = P2 - P1
        P13 = P3 - P1

        u = p12 / np.linalg.norm(p12)
        w = np.cross(u, p13)
        if np.linalg.norm(w) < 1e-5:
            raise Exception("The points lie almost on a straight line; no registration possible")
        w = w / np.linalg.norm(w)
        v = np.cross(w, u)

        U = P12 / np.linalg.norm(P12)
        W = np.cross(U, P13)
        if np.linalg.norm(W) < 1e-5:
            raise Exception("The points lie almost on a straight line; no registration possible")
        W = W / np.linalg.norm(W)
        V = np.cross(W, U)

        R_U_X = MakeBasis(U, V, W)
        r_u_x = MakeBasis(u, v, w)
        rotation = R_U_X.dot(r_u_x.T)

        m = float(((np.linalg.norm(P12) / np.linalg.norm(p12)) + (np.linalg.norm(P13) / np.linalg.norm(p13)) + (
        np.linalg.norm(P2 - P3) / np.linalg.norm(p2 - p3))) / 3)
        if m < 0.95 or m > 1.05:
            raise Exception("The transformation is not a rigid body transformation")

        X_S = 1 / 3 * (P1 + P2 + P3)
        x_s = 1 / 3 * (p1 + p2 + p3)
        translation = X_S - rotation.dot(x_s)

        return translation, rotation


    def Register(self):