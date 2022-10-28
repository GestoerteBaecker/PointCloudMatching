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
    table_indexes = dict()
    altered_order_indexes = list(range(number_of_points))
    shuffle(altered_order_indexes)
    altered_points = []
    for i in range(number_of_points):
        j = altered_order_indexes.index(i)
        altered_points.append(points[j] + RandomVector(max_deviation))
        table_indexes[j] = i
    return table_indexes, points, altered_points


# test additional points and deleted points
def Test(number_of_iterations, number_of_points, number_of_additional_points, number_of_points_to_delete, matching_threshold, max_deviation=0.0):
    lower_bound = -1000
    upper_bound = 1000
    for counter in range(number_of_iterations):
        expected_table_indexes, points, altered_points = ShuffleRandomPoints(number_of_points, matching_threshold, max_deviation)
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

        point_cloud = PointCloudMatching.PointCloud(points, matching_threshold)
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
    number_of_points = 20
    number_of_additional_points = 5
    number_of_points_to_delete = 3
    matching_threshold = 0.1
    max_deviation = matching_threshold/3

    success = Test(number_of_iterations, number_of_points, number_of_additional_points, number_of_points_to_delete, matching_threshold, max_deviation)
    if success:
        print("Pass")