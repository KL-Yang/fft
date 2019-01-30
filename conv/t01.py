import sys, unittest
import numpy as np
from scipy.linalg import toeplitz
sys.path.insert(0, './module')
import pytoep as tt

###assume they are normalized value around 1
def mat_close(a, b):
    if a.shape!=b.shape:
        print(a.shape)
        print(b.shape)
        print('shape different!')
        return False
    fa = a.flatten()
    fb = b.flatten()
    for i in range(len(fa)):
        if abs(fa[i]-fb[i])>1E-3:
            return False
    return True

def ref_mat04(a, f):
    assert len(a)==7
    ac = a[3:]      #1st column
    ar = a[:4]
    ar = ar[::-1]   #1st row
    ma = toeplitz(ac, ar)
    xb = np.dot(ma, f.reshape((4,1)))
    return xb.flatten()

class TestMat04Methods(unittest.TestCase):
    def test_001(self):
        xa  = np.array(range(1,8), dtype=np.float32)
        xf  = np.array(range(1,5), dtype=np.float32)
        xb  = tt.mat04(xa, xf)
        ref = ref_mat04(xa, xf)
        self.assertTrue(mat_close(xb, ref))

    def test_002(self):
        self.assertTrue(False)

if __name__ == '__main__':
    unittest.main()
    sys.exit(0)
