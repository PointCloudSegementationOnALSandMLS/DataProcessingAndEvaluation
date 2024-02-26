import math
import json
import utm

# Define the size of each grid cell in meters
cell_size = 110  # meters

# Define rotation angle in degrees
rotation_angle = 53  # degrees
rotation_angle_rad = math.radians(rotation_angle)  # convert rotation angle to radians
print(rotation_angle_rad)

# Define the starting point in UTM coordinates (x, y)
start_latitude = 40.7128  # example latitude (New York City)
start_longitude = -74.0060 # example UTM y-coordinate (meters)

# Determine UTM zone and hemisphere for the starting point
# Calculate rotated UTM coordinates for the grid cell
utm_x = 364983
utm_y =5699460
utm_zone = 32
utm_letter = "U"


# Generate rotated UTM grid cell coordinates and calculate corners
features = []
grid = 0
for i in range(11):  # 2500 meters in y direction (10 cells * 250 meters)
    for j in range(1):  # 2500 meters in x direction (10 cells * 250 meters)
        
        y_offset = [0,0,0 ]
        x_offset = [-165,-120,-65,-15,0,0,0,0,0,0,0,0,0]
        
        
            

        # Calculate coordinates of all corners for the grid cell
        bottom_left = utm.to_latlon(utm_x + (j * cell_size + x_offset[i]) * math.cos(rotation_angle_rad) - (i * cell_size + y_offset[j]) * math.sin(rotation_angle_rad),
                                    utm_y + (j * cell_size + x_offset[i]) * math.sin(rotation_angle_rad) + (i * cell_size + y_offset[j]) * math.cos(rotation_angle_rad),
                                    utm_zone, utm_letter)
        bottom_right = utm.to_latlon(utm_x + ((j+1) * cell_size + x_offset[i]) * math.cos(rotation_angle_rad) - (i * cell_size + y_offset[j]) * math.sin(rotation_angle_rad),
                                     utm_y + ((j+1) * cell_size + x_offset[i]) * math.sin(rotation_angle_rad) + (i * cell_size + y_offset[j]) * math.cos(rotation_angle_rad),
                                     utm_zone, utm_letter)
        top_right = utm.to_latlon(utm_x + ((j+1) * cell_size + x_offset[i]) * math.cos(rotation_angle_rad) - ((i+ 1) * cell_size + y_offset[j])* math.sin(rotation_angle_rad),
                                  utm_y + ((j+1) * cell_size + x_offset[i]) * math.sin(rotation_angle_rad) + ((i+1) * cell_size + y_offset[j]) * math.cos(rotation_angle_rad),
                                  utm_zone, utm_letter)
        top_left = utm.to_latlon(utm_x + (j * cell_size + x_offset[i]) * math.cos(rotation_angle_rad) - ((i +1) * cell_size + y_offset[j]) * math.sin(rotation_angle_rad),
                                 utm_y + (j * cell_size + x_offset[i]) * math.sin(rotation_angle_rad) + ((i +1) * cell_size + y_offset[j]) * math.cos(rotation_angle_rad),
                                 utm_zone, utm_letter)

        #utm_x = utm_x - 5 * math.cos(rotation_angle_rad) 
        #utm_y = utm_y - 5 * math.sin(rotation_angle_rad) 
        # Create a GeoJSON feature for the grid cell with latitude and longitude coordinates
        geometry = {
            "type": "Polygon",
            "coordinates": [[bottom_left[::-1], bottom_right[::-1], top_right[::-1], top_left[::-1], bottom_left[::-1]]],

        }
        feature = {
            "type": "Feature",
            "properties": {},
            "geometry": geometry
        }
        
        features.append(feature)
        grid +=1
    utm_x = utm_x + 10 * math.sin(rotation_angle_rad) # + 5 * math.cos(rotation_angle_rad) 
    utm_y = utm_y  - 10 * math.cos(rotation_angle_rad) # + 5 * math.sin(rotation_angle_rad)
# Create a GeoJSON feature collection with latitude and longitude coordinates
feature_collection = {
    "type": "FeatureCollection",
    "features": features
}

print(feature_collection)

# Save the GeoJSON data with latitude and longitude coordinates to a file
with open("grid_cells_lat_lon.geojson", "w") as geojson_file:
    json.dump(feature_collection, geojson_file)

print("GeoJSON data with Latitude and Longitude coordinates saved to grid_cells_lat_lon.geojson")
