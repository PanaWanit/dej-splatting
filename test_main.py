from lib.data_utils import read_all,read_points3D_colmap
from lib.gauss_model import GaussModel
from lib.point_utils import PointCloud

def testFunc():
    file_path = "../test/fern/images/"
    def testLoader():
        l = read_all(file_path + "images.txt", file_path + 'cameras.txt')
        print(len(l))
        print(l)
        pcd = read_points3D_colmap(file_path + "points3D.txt")
        print(len(pcd))
        print(pcd)
    
    def testGaussModel():
        gm = GaussModel()
        print(vars(gm),end='\n'*8)
        pcd = read_points3D_colmap(file_path + "points3D.txt")
        pcd = PointCloud(pcd[["X","Y","Z"]].to_numpy(),{
            "R":pcd["R"].to_numpy(),
            "G":pcd["G"].to_numpy(),
            "B":pcd["B"].to_numpy()
        })
        print(pcd,end='\n'*8)
        gm.create_from_pcd(pcd)
        print(vars(gm),end='\n'*8)

    
    # testLoader()

    testGaussModel()

testFunc()
