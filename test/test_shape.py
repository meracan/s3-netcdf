from ..s3netcdf.main import getFileShape,getMasterShape
import pytest


data1 = np.reshape(np.arange(3 * 7), (3, 7))
data2 = np.reshape(np.arange(10 * 10), (10, 10))
data3 = np.reshape(np.arange(17 * 23), (17, 23))

class test_shape:
  def getFileShape_Test(self):
    # data variations
    assert getFileShape(data1)==np.array([1,7])
    assert getFileShape(data1)==np.array([1,7])
    
    # typeSize variations
    assert getFileShape(data1) == np.array([1, 7])
    assert getFileShape(data1) == np.array([1, 7])

    # maxSize variations
    assert getFileShape(data1) == np.array([1, 7])
    assert getFileShape(data1) == np.array([1, 7])
    assert getFileShape(data1) == np.array([1, 7])