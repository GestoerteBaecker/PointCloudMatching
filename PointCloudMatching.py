import numpy as np


def FindKeyFromValueOfDict(map: dict, value: int) -> int:
    for key in map:
        value_in_dict = map[key]
        if value_in_dict == value:
            return key
    return None


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
        self.distances = PointCloud.CalculateDistanceMatrix(points)
        self.number_of_points = len(self.points)


    @staticmethod
    def CalculateDistanceMatrix(points: list) -> np.array:
        """
        Calculates the distance matrix of all points.
        :param points: list of np.arrays of the same dimension.
        :return: quadratic matrix of dimension len(points)
        """
        number_of_points = len(points)
        distances = np.zeros((number_of_points, number_of_points))
        for i in range(number_of_points):
            for j in range(i+1, number_of_points):
                if i == j:
                    continue
                distances[i, j] = np.linalg.norm(points[i] - points[j])
                distances[j, i] = distances[i, j]
        return distances


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


    def Assign(self, target_point_cloud: list) -> dict:
        """
        Finds the points in a second point cloud which correspond to the source point cloud
        :param target_point_cloud:
        :return: dictionary. 'key' is the index of the points from 'self.points'; 'value' is the index of the corresponding
        points from 'target_point_cloud'.
        """
        target_distances = PointCloud.CalculateDistanceMatrix(target_point_cloud)
        target_number_of_points = len(target_point_cloud)
        index_table = dict()
        altered_indexes_correspondencies_counter = dict()
        for i in range(self.number_of_points):
            correspondencies = [0] * target_number_of_points
            for j in range(self.number_of_points):
                source_distance = self.distances[i, j]
                target_indexes = self.FindDistance(target_distances, source_distance)
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
                        index_to_delete = FindKeyFromValueOfDict(index_table, altered_index)
                        del index_table[index_to_delete]
                        index_table[i] = altered_index
                        altered_indexes_correspondencies_counter[altered_index] = max(correspondencies)
                else:
                    index_table[i] = altered_index
                    altered_indexes_correspondencies_counter[altered_index] = max(correspondencies)

        # match points on basis of already matched points
        if len(index_table) == 0:
            return dict()
        index_added_point = list(index_table.keys())[0]
        altered_index_added_point = index_table[index_added_point]
        for i in range(self.number_of_points):
            if i in index_table.keys():
                continue
            distance_to_look_for = self.distances[i, index_added_point]
            altered_index = -1
            min_deviation = np.inf
            deviation = np.inf
            for j in range(target_number_of_points):
                if j in list(index_table.values()):
                    continue
                deviation = abs(distance_to_look_for - target_distances[altered_index_added_point, j])
                if deviation < min_deviation:
                    min_deviation = deviation
                    altered_index = j
            if altered_index == -1 or min_deviation > self.threshold:
                continue
            add_point = True
            for other_point_key in list(index_table.keys())[1:]:
                source_distance = self.distances[other_point_key, i]
                other_point_key_altered = index_table[other_point_key]
                target_distance = target_distances[altered_index, other_point_key_altered]
                if abs(source_distance - target_distance) > self.threshold:
                    add_point = False
                    break
            if add_point:
                index_table[i] = altered_index
        return index_table



### Test ###
from random import shuffle, uniform, randint

def ShuffleRandomPoints(number_of_points: int, lower_bound=-1000, upper_bound=1000) -> tuple:
    points = []
    for i in range(number_of_points):
        point = np.array([uniform(lower_bound, upper_bound), uniform(lower_bound, upper_bound), uniform(lower_bound, upper_bound)])
        points.append(point)
    table_indexes = dict()
    altered_order_indexes = list(range(number_of_points))
    shuffle(altered_order_indexes)
    altered_points = []
    for i in range(number_of_points):
        j = altered_order_indexes.index(i)
        altered_points.append(points[j])
        table_indexes[j] = i
    return table_indexes, points, altered_points


# test additional points and deleted points
def Test(number_of_iterations, number_of_points, number_of_additional_points, number_of_points_to_delete):
    lower_bound = -1000
    upper_bound = 1000
    for counter in range(number_of_iterations):
        expected_table_indexes, points, altered_points = ShuffleRandomPoints(number_of_points)
        for i in range(number_of_additional_points):
            point = np.array([uniform(lower_bound, upper_bound), uniform(lower_bound, upper_bound),
                              uniform(lower_bound, upper_bound)])
            index = randint(0, len(altered_points) - 1)
            for key in expected_table_indexes.keys():
                if expected_table_indexes[key] >= index:
                    expected_table_indexes[key] += 1
            altered_points.insert(index, point)
        i = 0
        while i < number_of_points_to_delete:
            index = randint(0, len(points) - 1)
            if not index in expected_table_indexes.keys():
                continue
            index_altered = expected_table_indexes[index]
            for key in expected_table_indexes.keys():
                if expected_table_indexes[key] > index_altered:
                    expected_table_indexes[key] -= 1
            del expected_table_indexes[index]
            altered_points.pop(index_altered)
            i += 1

        point_cloud = PointCloud(points, 0.1)
        actual_table_indexes = point_cloud.Assign(altered_points)
        if expected_table_indexes != actual_table_indexes:
            print("Fail on iteration", counter)
            print(expected_table_indexes)
            print(actual_table_indexes)
            print(points)
            print(altered_points)
            return False

        expected_table_indexes.clear()
        actual_table_indexes.clear()
        points.clear()
        altered_points.clear()

    return True

if __name__ == "__main__":
    number_of_iterations = 100
    number_of_points = 10
    number_of_additional_points = 5
    number_of_points_to_delete = 3

    success = Test(number_of_iterations, number_of_points, number_of_additional_points, number_of_points_to_delete)
    if success:
        print("Pass")