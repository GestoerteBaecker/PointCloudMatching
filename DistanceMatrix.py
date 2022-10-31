import numpy as np

class DistanceMatrix:
    """
    Combines points of a point cloud and all the distances between all points in an array as well as a list.
    This redundance could be removed in future versions.
    """
    def __init__(self, points: list, distances_matrix: np.array=None) -> None:
        """
        :param points: points of a point cloud
        :param distances_matrix: all distances between points of a point cloud as an array
        """
        self.points = points
        self.number_of_points = len(points)

        self.distances_matrix = np.array([])
        self.distances_list = []
        self.sorted_indices = []
        if distances_matrix is None:
            self.__CalculateDistanceMatrix__()
        else:
            self.__SetDistanceMatrix__(distances_matrix)


    def __getitem__(self, keys: tuple) -> float:
        """
        Gets the distance of the array.
        :param keys: point index 1 and point index 2 as tuple
        :return:
        """
        return self.distances_matrix[keys]


    def FindDistance(self, distance: float, equality_threshold: float) -> tuple:
        """
        Finds the given distance in the given distance matrix.
        :param distance_matrix: quadratic matrix.
        :param distance: distance which has to be find in the distance matrix.
        :return: if found, output is the indexes. If not found output is (-1,-1)
        """
        for i in range(self.number_of_points):
            for j in range(i+1, self.number_of_points):
                if abs(self.distances_matrix[i, j] - distance) < equality_threshold:
                    return i, j
        return -1, -1


    def GetPointIDsFromDistanceList(self, index_distance_list: int) -> tuple:
        """
        Converts the index of the distances saved as a list to the indices of the array representation.
        :param index_distance_list: Index of the distance list.
        :return: Indices of the distance array / matrix
        """
        if index_distance_list < 0 or index_distance_list >= len(self.distances_list):
            return -1, -1
        row = int(0.5 * (np.sqrt(8*index_distance_list+1)-1))+1
        column = int(index_distance_list - ((row-1)/2*(row)))
        return row, column


    @classmethod
    def FindSubsetOfDistanceMatrixWithMaximumDistances(cls, points: list, number_of_points_to_use: int):
        """
        Gets a DistanceMatrix-Object with a reduced number of points. The points are chosen by their impact on the
        largest distances.
        CAUTION: The order of the points is not guaranteed to remain unaltered!
        :param points: all points possible
        :param number_of_points_to_use: number hoe many points should be chosen
        :return: DistanceMatrix
        """
        all_point_distance_matrix = cls(points)
        if all_point_distance_matrix.number_of_points <= number_of_points_to_use:
            return all_point_distance_matrix

        old_point_ids_added = []
        distance_index = len(all_point_distance_matrix.sorted_indices) -1
        while len(old_point_ids_added) < number_of_points_to_use:
            point_id_1, point_id_2 = all_point_distance_matrix.GetPointIDsFromDistanceList(all_point_distance_matrix.sorted_indices[distance_index])
            if not point_id_1 in old_point_ids_added:
                old_point_ids_added.append(point_id_1)
                if len(old_point_ids_added) == number_of_points_to_use:
                    break
            if not point_id_2 in old_point_ids_added:
                old_point_ids_added.append(point_id_2)
            distance_index -= 1
        old_point_ids_added.sort()

        add_index = 0
        points_reduced = []
        distance_matrix_reduced = np.zeros((number_of_points_to_use, number_of_points_to_use))
        for i, old_ind_1 in enumerate(old_point_ids_added):
            point = points.pop(old_ind_1)
            points.insert(add_index, point)
            add_index += 1
            points_reduced.append(point)
            for j in range(i):
                old_ind_2 = old_point_ids_added[j]
                distance_matrix_reduced[i, j] = all_point_distance_matrix[old_ind_1, old_ind_2]
                distance_matrix_reduced[j, i] = distance_matrix_reduced[i, j]

        ret_val = DistanceMatrix(points_reduced, distance_matrix_reduced)
        return ret_val


    def __GetSortedIndices__(self, distances: list) -> list:
        """
        Do not use this function separately.
        Get the indices of points as they were if they were ordered by value.
        :param distances: Distances as list
        :return: indices ordered list
        """
        return [key_val[0] for key_val in sorted(enumerate(distances), key=lambda x: x[1])]


    def __SetDistanceMatrix__(self, distances_matrix: np.array) -> None:
        """
        Do not use this function separately.
        Sets the distance array / matrix separately.
        :param distances_matrix: distances between all points.
        """
        self.distances_matrix = distances_matrix
        self.distances_list = []
        for i in range(self.number_of_points):
            for j in range(i):
                self.distances_list.append(self.distances_matrix[i, j])
        self.sorted_indices = self.__GetSortedIndices__(self.distances_list)


    def __CalculateDistanceMatrix__(self) -> np.array:
        """
        Do not use this function separately.
        Calculates the distance matrix of all points.
        :return: quadratic matrix of dimension len(points) and the maximum distance between two points of the cloud
        """
        self.distances_matrix = np.zeros((self.number_of_points, self.number_of_points))
        self.distances_list = []
        for i in range(self.number_of_points):
            for j in range(i):
                distance = np.linalg.norm(self.points[i] - self.points[j])
                self.distances_matrix[i, j] = distance
                self.distances_matrix[j, i] = distance
                self.distances_list.append(distance)
        self.sorted_indices = self.__GetSortedIndices__(self.distances_list)
