# PointCloudMatching
Used to match smaller point clouds to each other. The more points are given, the longer the computation time takes. One purpose would be a geodetic measurement of marked points.

## Use conditions
- any distance between any two arbitrary points from the point cloud has to be unique up to the given threshold
- the inner geometry (e.g. the distance between all points) of the point clouds to match must be conserved up to the given threshold (e.g. only rigid body transformation allowed)

## Use
Instantiate an object of this class by giving a point cloud. Match / assign a second point cloud by using the method 'Assign'. The output is a dictionary which maps points of the source point cloud to points of the target / second point cloud. Example: {0: 2, 1:1, 2:0} The point 0 of the source point cloud is equal to the point 2 of the target point cloud.
