from DistanceMatrix import *
from PointCloudFitting import *


class PointCloud:
    """
    PointCloud.
    Used to match smaller point clouds to each other. The inner geometry (distances) of these two clouds has to be the same
    (rigid body transformation, but with a certain threshold of change of inner geometry).
    One purpose would be a geodesic measurement of marked points.

    Use: Instantiate an object of this class by giving a point cloud. Match / assign a second point cloud by using the method
    'Assign'. The output is a dictionary which maps points of the source point cloud to points of the target / second point cloud.
    Example: {0: 2, 1:1, 2:0} The point 0 of the source point cloud is equal to the point 2 of the target point cloud.

    CAUTION: The order of the points of the second point cloud is not guaranteed to remain unaltered!
    """

    def __init__(self, points: list, threshold: float) -> None:
        """
        :param points: list of np.arrays of the same dimension.
        :param threshold: defines the maximum deviation of two distances to be identified as equal
        """
        self.points = points
        self.threshold = threshold
        self.distances = DistanceMatrix(points)
        self.number_of_points = len(self.points)
        self.points_to_match = None
        self.distances_to_match = None
        self.transformation = None
        self.assigned = False
        self.index_table = dict()


    def Preassign(self, target_point_cloud: list, expected_blunder_ratio: float=0.0) -> dict:
        """
        Finds the points in a second point cloud which correspond to the source point cloud. These correspondencies are
        only calculated for 10 points or an expected number of blunder points (+3 for calculating the registration) due to
        performance reasons.
        CAUTION: The order of the points of the second point cloud is not guaranteed to remain unaltered!
        :param target_point_cloud:
        :param expected_blunder_ratio: ratio of blunders to real number of points
        :return: dictionary. 'key' is the index of the points from 'self.points'; 'value' is the index of the corresponding
        points from 'target_point_cloud'.
        """
        if expected_blunder_ratio < 0:
            expected_blunder_ratio = 0
        elif expected_blunder_ratio > 1:
            expected_blunder_ratio = 1
        self.points_to_match = target_point_cloud
        target_number_of_all_points = len(target_point_cloud)
        expected_blunders = target_number_of_all_points * expected_blunder_ratio
        target_number_of_points = min([max([expected_blunders+3, 10]), target_number_of_all_points])
        self.distances_to_match = DistanceMatrix.FindSubsetOfDistanceMatrixWithMaximumDistances(self.points_to_match, target_number_of_points)
        self.index_table = dict()
        altered_indexes_correspondencies_counter = dict()
        for i in range(self.number_of_points):
            correspondencies = [0] * target_number_of_points
            for j in range(self.number_of_points):
                source_distance = self.distances[i, j]
                target_indexes = self.distances_to_match.FindDistance(source_distance, self.threshold)
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


    def Assign(self, target_point_cloud: list = list(), expected_blunder_ratio: float=0.0) -> dict:
        """
        Finds the points in a second point cloud which correspond to the source point cloud.
        :param target_point_cloud: (Measured) Points whose inner geometry corresponds to the one of the first inserted
        point cloud.
        :param expected_blunder_ratio: ratio of blunders to real number of points
        :return: dictionary. 'key' is the index of the points from 'self.points'; 'value' is the index of the corresponding
        points from 'target_point_cloud'.
        """
        if len(target_point_cloud) != 0:
            self.Preassign(target_point_cloud, expected_blunder_ratio)
            if len(self.index_table) == len(target_point_cloud):
                return self.index_table

        self.transformation = PointCloudFitting(self.points, self.distances_to_match.points, self.index_table)
        try:
            self.transformation.StartAdjustment()
        except Exception as ex:
            raise Exception("There are not enough points or the geometry of the distribution is invalid. Internal Error: " + \
                            str(ex))
        self.__AssignByCoordinates__()
        self.assigned = True
        return self.index_table


    def CalculateTransformation(self) -> PointCloudFitting:
        """
        Calculates the transformation parameters. Those are given by the PointCloudFitting-object which has the attributes 'translation'
        and 'rotation' and function 'Transform' for transform between the systems.
        :return: PointCloudFitting object
        """
        if not self.assigned:
            raise Exception("Please first match a point cloud to the first one by using the function 'Assign'")
        temp = PointCloudFitting(self.points, self.points_to_match, self.index_table)
        temp.StartAdjustment()
        return temp


    def __AssignByCoordinates__(self) -> None:
        """
        Do not use this function separately.
        Matches points based on the their position in space.
        """
        if self.transformation is None:
            raise Exception("Please call 'Assign' instead")
        for i_first, point in enumerate(self.points):
            if i_first in self.index_table.keys():
                continue
            point_first = self.transformation.Transform(point)
            for i_second, point_second in enumerate(self.points_to_match):
                if i_second in self.index_table.values():
                    continue
                x_equals = abs(point_first[0] - point_second[0]) < self.threshold
                y_equals = abs(point_first[1] - point_second[1]) < self.threshold
                z_equals = abs(point_first[2] - point_second[2]) < self.threshold
                if x_equals and y_equals and z_equals:
                    self.index_table[i_first] = i_second
                    break
