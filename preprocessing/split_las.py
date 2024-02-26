import laspy
import geojson
from shapely.geometry import shape, Polygon, Point
import utm
import numpy as np


            
            
def split_las_by_grids(input_las_path, geojson_path):
    # Read bounding boxes from GeoJSON file
    with open(geojson_path, 'r') as geojson_file:
        geojson_data = geojson.load(geojson_file)
        
    polygons = [None]*8
    for idx, feature in enumerate(geojson_data['features']):
            polygon_geojson =feature['geometry']['coordinates'][0]
            print(polygon_geojson)
            polygon_tuple = tuple(map(tuple, polygon_geojson))
            print(polygon_tuple)
            polygons[idx] = Polygon(polygon_tuple)
                
    
    with laspy.open(input_las_path, mode='r') as lasfile:
            savedPoints = 0
            for point in lasfile.chunk_iterator(1):
                    if point['Z'][0] > -3000 and point['Z'][0] <50000:
                        latlng_point = utm.to_latlon(point['x'][0], point['y'][0], 32, "U")
                        test_point = Point(latlng_point[1], latlng_point[0])
                        for idx, polygon in enumerate(polygons):
                            if polygon.contains(test_point):
                                savedPoints +=1
                                if(savedPoints %10000 == 0):
                                    print(savedPoints)
                                output_las_path = f"Essen/mobile/grid_{idx}.las"
                                with laspy.open(output_las_path, mode='w', header=lasfile.header) as output_las:
                                    output_las.write_points(point)


# Example usage
input_las_path = '/mnt/c/Users/nick1/Downloads/20230605_Abgabe_Mauern_aus_Punktwolken_per_KI/R_20220629_west(2)-000_LAS1_2.las'  # Path to the input LAS file
geojson_path = 'grid_cells_lat_lon.geojson'  # Path to the GeoJSON file containing bounding boxes

split_las_by_grids(input_las_path, geojson_path)
