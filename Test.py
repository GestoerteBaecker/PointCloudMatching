import PointCloudMatching
import numpy as np

### Test ###
from random import shuffle, uniform, randint

def RandomVector(magnitude:float) -> np.array:
    vector = np.array([uniform(-1, 1), uniform(-1, 1), uniform(-1, 1)])
    magnitude = uniform(0, magnitude)
    vector = vector * (magnitude / np.linalg.norm(vector))
    return vector

def ShuffleRandomPoints(number_of_points: int, matching_threshold, max_deviation=0.0, lower_bound=-1000, upper_bound=1000) -> tuple:
    points = []
    for i in range(number_of_points):
        point = np.array([uniform(lower_bound, upper_bound), uniform(lower_bound, upper_bound), uniform(lower_bound, upper_bound)])
        add_point = True
        for j in range(len(points)):
            if np.linalg.norm(point - points[j]) < matching_threshold:
                add_point = False
                break
        if add_point:
            points.append(point)
    altered_order_indexes = list(range(number_of_points))
    shuffle(altered_order_indexes)
    altered_points = []
    for i in range(number_of_points):
        j = altered_order_indexes.index(i)
        altered_points.append(points[j] + RandomVector(max_deviation))
    return points, altered_points


def PointsAreEqual(first_points, second_points, index_table, max_error):
    for i in index_table.keys():
        if np.linalg.norm(first_points[i] - second_points[index_table[i]]) > max_error:
            return False
    return True


# test additional points and deleted points
def Test(number_of_iterations, number_of_points, number_of_additional_points, number_of_points_to_delete, matching_threshold, max_deviation=0.0):
    lower_bound = -1000
    upper_bound = 1000
    for counter in range(number_of_iterations):
        points, altered_points = ShuffleRandomPoints(number_of_points, matching_threshold, max_deviation)
        for i in range(number_of_additional_points):
            point = np.array([uniform(lower_bound, upper_bound), uniform(lower_bound, upper_bound),
                              uniform(lower_bound, upper_bound)])
            index = randint(0, len(altered_points) - 1)
            altered_points.insert(index, point)
        for i in range(number_of_points_to_delete):
            index = randint(0, len(altered_points) - 1)
            altered_points.pop(index)

        point_cloud = PointCloudMatching.PointCloud(points, matching_threshold)
        actual_table_indexes = point_cloud.Assign(altered_points)

        if len(actual_table_indexes) < number_of_points - number_of_points_to_delete:
            print("Some points are not matched")
            return False

        if not PointsAreEqual(points, altered_points, actual_table_indexes, max_deviation):
            print("Points are incorrectly matched")
            print("Fail on iteration", counter)
            return False

        actual_table_indexes.clear()
        points.clear()
        altered_points.clear()

    print("Pass")
    return True

if __name__ == "__main__":
    number_of_iterations = 100
    number_of_points = 30
    number_of_additional_points = 5
    number_of_points_to_delete = 3
    matching_threshold = 0.1
    max_deviation = matching_threshold/3

    success = Test(number_of_iterations, number_of_points, number_of_additional_points, number_of_points_to_delete, matching_threshold, max_deviation)
