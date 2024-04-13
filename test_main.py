from lib.data_utils import read_all
from lib.gauss_model import GaussModel

def testFunc():
    def testLoader():
        file_path = "../test/fern/images/"
        l = read_all(file_path + "images.txt", file_path + 'cameras.txt')
        print(len(l))
        print(l)
    
    def testGaussModel():
        gm = GaussModel()
        print(gm.__dict__)
    
    # testLoader()

    testGaussModel()

testFunc()
