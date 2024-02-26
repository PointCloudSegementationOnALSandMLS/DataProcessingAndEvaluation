
import laspy
import open3d as o3d
import numpy as np
import open3d.ml.torch as ml3d  # just switch to open3d.ml.tf for tf usage
import open3d.ml as _ml3d
import os
from os.path import exists, join, dirname 
import logging
import json
import matplotlib.pyplot as plt
import Open3DML.ml3d.datasets.essen as essen
from plyfile import PlyData
log = logging.getLogger(__name__)

def dataInsp():
    las = laspy.read('/mnt/c/Users/nick1/Downloads/20230605_Abgabe_Mauern_aus_Punktwolken_per_KI/L_20220629_west(2)-000_LAS1_2.las')
    print(list(las.point_format.dimension_names))
    #plt.rcParams['agg.path.chunksize'] = 10000
    #plt.plot(las.y,las.z)
    #plt.show()
    #point_data = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))
    #new_array = point_data[:10000000, :]
    x_scale = las.header.scale[0]
    y_scale = las.header.scale[1]
    z_scale = las.header.scale[2]
    x_offset = las.header.offset[0]
    y_offset = las.header.offset[1]
    z_offset = las.header.offset[2]

    # Access the x, y, z coordinates of the unscaled points
    x_unscaled = las.X
    y_unscaled = las.Y
    z_unscaled = las.Z

# Convert unscaled points to real-world coordinates and height
    x_real = (x_unscaled * x_scale) + x_offset
    y_real = (y_unscaled * y_scale) + y_offset
    z_real = (z_unscaled * z_scale) + z_offset
    print(x_real)
    point_data = np.stack([x_real, y_real, z_real], axis=0).transpose((1, 0))
    print(point_data)
    new_array = point_data[:10000000, :]
    geom = o3d.geometry.PointCloud()
    geom.points = o3d.utility.Vector3dVector(new_array)
    o3d.visualization.draw_geometries([geom])
    
def dataInsp2():
    las = laspy.read('/mnt/c/Users/nick1/Downloads/20230605_Abgabe_Mauern_aus_Punktwolken_per_KI/L_20190902(20)_LAS1_2 (3).las')
    #print(list(las.point_format.dimension_names))
    #plt.rcParams['agg.path.chunksize'] = 10000
    #plt.plot(las.X, las.Y)
    #plt.show()
    point_data = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))
    print(point_data)
    geom = o3d.geometry.PointCloud()
    geom.points = o3d.utility.Vector3dVector(point_data)
    o3d.visualization.draw_geometries([geom])

def dataprep():
    las = laspy.read('/mnt/c/Users/nick1/Downloads/20230605_Abgabe_Mauern_aus_Punktwolken_per_KI/L_20220629_west(2)-000_LAS1_2.las')
    las
    print(list(las.point_format.dimension_names))
    #print(list(las.X))
    point_data = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))
    #print(point_data)
    new_array = point_data[:10000000, :]
    #print(new_array)
    #plt.plot(new_array[:,0],new_array[:,2])
    #plt.show()
    filter = lambda row: row[2]> -3000 and row[2]< 50000

    #create boolean mask by applying filter
    mask = np.apply_along_axis(filter, 1, new_array)
    #print(mask)
    new_array = new_array[mask]
    #plt.plot(new_array[:,0],new_array[:,2])
    #plt.show()
    print(new_array)
    #geom = o3d.geometry.PointCloud()
    #geom.points = o3d.utility.Vector3dVector(new_array)
    #o3d.visualization.draw_geometries([geom])
    np.save("points_show", new_array)
def shrinkData():
    my_array= np.load("points_show.npy")
    print(my_array.shape)
    new_array = my_array[:3000000, :]

    # Verify the shape of the new array
    #print(new_array)
    np.save("points_show", new_array)
    labels = np.zeros(3000000, dtype =int)
    np.save("labels_show", labels)
    geom = o3d.geometry.PointCloud()
    geom.points = o3d.utility.Vector3dVector(new_array)
    o3d.visualization.draw_geometries([geom])

def readDales():
    class_counts = {}

    for f in os.listdir('/mnt/c/Users/nick1/Downloads/dales_semantic_segmentation_ply.tar/dales_semantic_segmentation_ply/dales_ply/train'):
    # Load the PLY file
        print(f)
        plydata = PlyData.read('/mnt/c/Users/nick1/Downloads/dales_semantic_segmentation_ply.tar/dales_semantic_segmentation_ply/dales_ply/train/' + f)
        #print(plydata)
        
        # Initialize a dictionary to store class counts
        

        # Assuming the class labels are stored in the 'class_label' property
        class_property = 'class'

        # Iterate through the data and count class occurrences
        for vertex in plydata['vertex']:
            class_label = vertex[class_property]
            if class_label in class_counts:
                class_counts[class_label] += 1
            else:
                class_counts[class_label] = 1

    # Print the class counts
    for class_label, count in class_counts.items():
        print(f"Class {class_label}: {count} occurrences")

def main():
    new_array= np.load("points_show.npy")
    labels = np.load("labels_show.npy")
    feat = np.zeros(new_array.shape, dtype=np.float32)
    data = {
                'name': "essen",
                'point': new_array,
                'feat': feat,
                'label': labels,
            }   
    semantic_labels = ml3d.datasets.Semantic3D.get_label_to_names()
    v = ml3d.vis.Visualizer()
    lut = ml3d.vis.LabelLUT()
    for val in sorted(semantic_labels.keys()):
        lut.add_label(semantic_labels[val], val)
    v.set_lut("labels", lut)
    v.set_lut("pred", lut)

    cfg_file= './Open3D-ML/ml3d/configs/randlanet_toronto3d.yml'
    ckpt_path_r = "./randlanet_toronto.pth"
    cfg = _ml3d.utils.Config.load_from_file(cfg_file)
    cfg.model.ckpt_path = ckpt_path_r
    model = ml3d.models.RandLANet(**cfg.model)
    print(cfg)
    #kpconv_url = "https://storage.googleapis.com/open3d-releases/model-zoo/kpconv_semantickitti_202009090354utc.pth"
    
    print(model.cfg.ckpt_path)
    pipeline_r = ml3d.pipelines.SemanticSegmentation(model, **cfg.pipeline)
    pipeline_r.load_ckpt(model.cfg.ckpt_path)
    results_r = pipeline_r.run_inference(data)
    pred_label_r = (results_r['predict_labels'] + 1).astype(np.int32)
    # Fill "unlabeled" value because predictions have no 0 values.
    pred_label_r[0] = 0
    vis_d = {
                'name': "essen",
                "points": data['point'],
                "labels": data['label'],
                "pred": pred_label_r,
            }
    vis_js = {
        'name': "essen",
        "points": vis_d["points"].tolist(),
        "labels": vis_d["labels"].tolist(),
        "pred": vis_d["pred"].tolist(),
    }
    json.dump( vis_js, open( "predicted_show.json", 'w' ) )

    las = laspy.read('/mnt/c/Users/nick1/Downloads/20230605_Abgabe_Mauern_aus_Punktwolken_per_KI/L_20220629_west(2)-000_LAS1_2.las')
    #print(list(las.point_format.dimension_names))
    #plt.rcParams['agg.path.chunksize'] = 10000
    #plt.plot(las.y,las.z)
    #plt.show()
    #point_data = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))
    #new_array = point_data[:10000000, :]
    point_data = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))
    vis_scaledPoints = {
        'name': "essen",
        "points": point_data.tolist(),
        "labels": vis_d["labels"].tolist(),
        "pred": vis_d["pred"].tolist(),
    }
    json.dump( vis_js, open( "predicted_show.json", 'w' ) )
    json.dump( vis_scaledPoints, open( "predicted_show_scaled.json", 'w' ) )
    #v.visualize([vis_d])


def vis():
    #pointsNumpy = np.load("grid_1.npy", allow_pickle=True)
    #print(pointsNumpy)
    semantic_labels = {
            0: 'Unclassified',
            1: 'Natural',
            2: 'Building',
            3: 'Pole',
            4: 'Street',
            5: 'Car',
            6: 'Wall',
            7: 'Fence',
        }
    v = ml3d.vis.Visualizer()
    lut = ml3d.vis.LabelLUT()
    
    for val in sorted(semantic_labels.keys()):
        lut.add_label(semantic_labels[val], val)
    v.set_lut("labels", lut)
    v.set_lut("pred", lut)
    data = json.load( open( "predicted_essen_mobile_200.json" ) )  

    #points = data["points"]
    #print(points)
    #npPoints= np.array(points)
    #print(points[0])
    #print(npPoints[:,0])
    #print(np.min(npPoints[:,0]))
    #offsets= np.min(npPoints, axis= 0)
    #normPoints= npPoints -offsets
    #print(normPoints)
    #normPoints = normPoints / [0.1, 0.1, 0.1]
    #print(normPoints)

    #data["points"] = normPoints

    #geom = o3d.geometry.PointCloud()

    #geom.points = o3d.utility.Vector3dVector(data)
    #o3d.visualization.draw_geometries([geom])
    #points = data["points"]
    #las = laspy.read('/mnt/c/Users/nick1/Downloads/20230605_Abgabe_Mauern_aus_Punktwolken_per_KI/L_20220629_west(2)-000_LAS1_2.las')
   # x_scale = las.header.scale[0]
#    y_scale = las.header.scale[1]
 #   z_scale = las.header.scale[2]
  #  x_offset = las.header.offset[0]
   # y_offset = las.header.offset[1]
   # z_offset = las.header.offset[2]



    #print(points[0:])
    #print(x_scale)
# Convert unscaled points to real-world coordinates and height
    #points[0] = ( np.asarray(points[0:]) * x_scale) + x_offset
    #points[1] = ( np.asarray(points[1]) * y_scale) + y_offset
    #points[2] = ( np.asarray(points[2]) * z_scale) + z_offset

    


    
    #plt.plot(points[0], points[2])
    #plt.show()
    #print(points[0])

    #unten= np.quantile(points[0], 0.01)
    #oben = np.quantile(points[0], 0.99)
    #print(np.max(points[0]))
    #print(unten)
    #print(oben)
    #filter = lambda row: row[0]> unten and row[0]<  oben

    #create boolean mask by applying filter
    #mask = np.apply_along_axis(filter, 1, points)
    #print(mask)
    #print(points)
    #points = np.array(points)[mask]
    #print(points)


    #labels = np.zeros(len(points[0]), dtype =int)


    #data["points"] = points
    #data["labels"] = labels
    #print(data["points"])
    v.visualize([data])

def visKitty():
    dataset = ml3d.datasets.SemanticKITTI(dataset_path='/mnt/d/Punkte')

# get the 'all' split that combines training, validation and test set
    all_split = dataset.get_split('all')

    # print the attributes of the first datum
    print(all_split.get_attr(0))

    # print the shape of the first point cloud
    print(all_split.get_data(0)['point'].shape)
    print(all_split.get_data(0)['point'])
    plt.plot(all_split.get_data(0)['point'][:,0],all_split.get_data(0)['point'][:,2])
    plt.show()

    # show the first 100 frames using the visualizer
    vis = ml3d.vis.Visualizer()
    vis.visualize_dataset(dataset, 'all', indices=range(50))


def readToronto():
    pcd = o3d.io.read_point_cloud("./Toronto_3D/L001.ply")
    print(pcd)
    xyz_load = np.asarray(pcd.points)
    print('xyz_load')
    print(xyz_load)
    #o3d.visualization.draw_geometries([pcd])
    plt.rcParams['agg.path.chunksize'] = 10000
    plt.plot(xyz_load[:,0],xyz_load[:,2])
    plt.show()

def las_to_numpy(input_file):
    las = laspy.read(input_file)
    points = np.stack([las.x, las.y, las.z, las.intensity], axis=0).transpose((1, 0))
    print(points)
    return points

def numpy_to_ply(points, output_file):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:,:3]) 
    o3d.io.write_point_cloud(output_file, pcd)

def numpy_to_o3dml(points, output_file):
    labels = np.zeros((points.shape[0],), dtype=np.int32)
    data = {'point': points[:,:3], 'feat': points[:,3], 'label': labels}
    np.save(output_file, data, allow_pickle=True)


def splitLas():
    for i in range(0,11):
        input_file = "Daten_Essen/Allee/grids/mobile/L/grid_{}/grid_{}.las".format(i,i)
        print(input_file)
        path1= "Daten_Essen/Allee/grids/mobile/L/grid_{}/split0".format(i)
        path2 = "Daten_Essen/Allee/grids/mobile/L/grid_{}/split1".format(i)
        path3 = "Daten_Essen/Allee/grids/mobile/L/grid_{}/split2".format(i)
        path4 = "Daten_Essen/Allee/grids/mobile/L/grid_{}/split3".format(i)
        if not os.path.exists(path1):
            os.makedirs(path1)
        if not os.path.exists(path2):
            os.makedirs(path2)
        if not os.path.exists(path3):
            os.makedirs(path3)
        if not os.path.exists(path4):
            os.makedirs(path4)
        output_file1 = "Daten_Essen/Allee/grids/mobile/L/grid_{}/split0/grid_{}_split0.ply".format(i,i)
        output_file2 = "Daten_Essen/Allee/grids/mobile/L/grid_{}/split1/grid_{}_split1.ply".format(i,i)
        output_file3 = "Daten_Essen/Allee/grids/mobile/L/grid_{}/split2/grid_{}_split2.ply".format(i,i)
        output_file4 = "Daten_Essen/Allee/grids/mobile/L/grid_{}/split3/grid_{}_split3.ply".format(i,i)

        # Convert LAS to NumPy
        las_points = las_to_numpy(input_file)
        #las_points = np.array([[1,2],[3,4],[5,6],[7,8],[9,10]])


        # Split based on condition
        part1, part2, part3, part4 = np.array_split(las_points, 4, axis=0)
        print("part1")
        print (part1)
        print("part4")
        print(part4)

        # Convert NumPy back to LAS
        numpy_to_o3dml(part1, output_file1)
        numpy_to_o3dml(part2, output_file2)
        numpy_to_o3dml(part3, output_file3)
        numpy_to_o3dml(part4, output_file4)
        #numpy_to_las(las_points_part2, output_file1, las_header)

def convert_las_to_npy(las_file, npz_file):
    # Open the LAS file for reading
    las_data = laspy.file.File(las_file, mode='r')

    # Extract point cloud data
    x = las_data.x
    y = las_data.y
    z = las_data.z

    # Assuming 'classification' is the class attribute in the LAS file (you may need to adjust this)
    if 'classification' in las_data.point_format.lookup.keys():
        point_class = las_data.classification
    else:
        point_class = np.zeros_like(x)  # Default to all zeros if 'classification' is missing

    # Assuming additional features are present in the LAS file as 'feat_1', 'feat_2', etc.
    feat_attributes = []
    for i in range(1, las_data.point_format.num_extra_bytes + 1):
        feat_name = f'feat_{i}'
        feat_data = las_data[f'extra_byte_{i}']
        feat_attributes.append(feat_data)

    # Stack all attributes into a single array
    point_cloud = np.column_stack((x, y, z, point_class, *feat_attributes))

    # Save the point cloud as an NPZ file
    np.savez(npz_file, point_cloud=point_cloud)

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(asctime)s - %(module)s - %(message)s",
    )

    splitLas()