import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils


def dense_crf(img, output_probs):
    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    img = np.ascontiguousarray(img)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=70, srgb=5, rgbim=img, compat=5)

    Q = d.inference(10)
    Q = np.array(Q).reshape((c, h, w))
    return Q
