import numpy as np
import obj_io
from scipy.spatial.transform import Rotation
import os
import cv2

# https://gist.github.com/davegreenwood/e1d2227d08e24cc4e353d95d0c18c914


def triangulate_nviews_1point(P, ip):
    """
    Triangulate a point visible in n camera views.
    P is a list of camera projection matrices.
    ip is a list of homogenised image points. eg [ [x, y, 1], [x, y, 1] ], OR,
    ip is a 2d array - shape nx3 - [ [x, y, 1], [x, y, 1] ]
    len of ip must be the same as len of P
    """
    if not len(ip) == len(P):
        raise ValueError('Number of points and number of cameras not equal.')
    ip = np.array(ip)
    if len(ip.shape) != 2:
        raise ValueError("Shape must be 2")
    n, dim = ip.shape
    if dim == 3:
        pass
    elif dim == 2:
        ones = np.ones((n, 3))
        ones[..., :2] = ip
        ip = ones
    else:
        raise ValueError('Length must be 2 or 3')
    M = np.zeros([3*n, 4+n])
    for i, (x, p) in enumerate(zip(ip, P)):
        M[3*i:3*i+3, :4] = p
        M[3*i:3*i+3, 4+i] = -x
    V = np.linalg.svd(M)[-1]
    X = V[-1, :4]
    return (X / X[3])[..., :3]


def triangulate_2views_1point(P1, P2, x1, x2):
    """
    Two-view triangulation of points in
    x1,x2 (nx3 homog. coordinates).
    Similar to openCV triangulatePoints.
    """
    if not len(x2) == len(x1):
        raise ValueError("Number of points don't match.")
    X = [triangulate_nviews_1point([P1, P2], [x[0], x[1]])
         for x in zip(x1, x2)]
    return np.array(X)


def triangulate_nviews(P, ip):
    _, LMK_NUM, _ = ip.shape
    points = []
    for i in range(LMK_NUM):
        ipp = ip[:, i]
        p = triangulate_nviews_1point(P, ipp)
        points.append(p)
    return np.array(points)


def evalReprojectionError(points3d, lmk2ds, w2c_Rs, w2c_ts, Ks, weight=None):
    VIEW_NUM, LMK_NUM, _ = lmk2ds.shape
    rotated = np.einsum("ijk,lk->ilj", w2c_Rs, points3d)
    translated = rotated + w2c_ts[:, None]
    xs = Ks[:, 0, 0, None] / translated[..., 2] * \
        translated[..., 0] + Ks[:, 0, 2, None]
    ys = Ks[:, 1, 1, None] / translated[..., 2] * \
        translated[..., 1] + Ks[:, 1, 2, None]
    projected = np.stack([xs, ys], axis=-1)
    reprojection_diff = projected - lmk2ds
    # (VIEW_NUM, LMK_NUM)
    reprojection_error = np.sqrt(
        np.sum(reprojection_diff * reprojection_diff, axis=-1))
    if weight is not None:
        reprojection_error = reprojection_error * weight
    reprojection_error_view = np.sum(reprojection_error, axis=-1) / LMK_NUM
    return projected, reprojection_error, reprojection_error_view


def triangulateLandmarkLeastSquares(lmk2ds, w2c_Rs, w2c_ts, Ks):
    VIEW_NUM = len(lmk2ds)

    Rt = np.eye(3, 4)[None, ...].repeat(VIEW_NUM, axis=0)
    Rt[..., 0:3, 0:3] = w2c_Rs
    Rt[..., 0:3, 3] = w2c_ts
    proj_mats = np.matmul(Ks, Rt)

    points = triangulate_nviews(proj_mats, lmk2ds)
    return points


def triangulateLandmarkRANSAC(lmk2ds, w2c_Rs, w2c_ts, Ks,
                              inlier_th, max_iter=1000):
    SELECT_NUM = 2
    VIEW_NUM = len(lmk2ds)
    best_inlier_mask = np.zeros(VIEW_NUM, dtype=bool)
    best_rprj_view = np.ones(VIEW_NUM) * np.inf
    best_points = None
    best_projected = None
    best_hypothesis = None

    Rt = np.eye(3, 4)[None, ...].repeat(VIEW_NUM, axis=0)
    Rt[..., 0:3, 0:3] = w2c_Rs
    Rt[..., 0:3, 3] = w2c_ts
    proj_mats = np.matmul(Ks, Rt).astype(np.float32)

    lmk2ds = lmk2ds.astype(np.float32)

    iter = 0
    while max_iter > iter:
        iter += 1
        selected = np.random.randint(0, VIEW_NUM, (SELECT_NUM))
        while len(set(selected)) != SELECT_NUM:
            selected = np.random.randint(0, VIEW_NUM, (SELECT_NUM))
        rnd_mask = np.zeros(VIEW_NUM, dtype=bool)
        rnd_mask[selected] = True

        proj_rnd, lmk2d_rnd = proj_mats[rnd_mask], lmk2ds[rnd_mask]

        points = cv2.triangulatePoints(
            proj_rnd[0], proj_rnd[1], lmk2d_rnd[0].T, lmk2d_rnd[1].T)
        points = points.T
        points = points[..., :3] / points[..., 3][..., None]

        projected, rprj_err, rprj_err_view = evalReprojectionError(
            points, lmk2ds, w2c_Rs, w2c_ts, Ks)
        inlier_mask = rprj_err_view < inlier_th
        inlier_num = inlier_mask.sum()
        if inlier_num < 1:
            continue
        if inlier_num < best_inlier_mask.sum():
            continue
        if inlier_mask.sum() == best_inlier_mask.sum() and\
                rprj_err_view[inlier_mask].mean() > best_rprj_view[best_inlier_mask].mean():
            continue

        best_inlier_mask = inlier_mask
        best_rprj_view = rprj_err_view
        best_points = points
        best_projected = projected
        best_hypothesis = selected

    return best_inlier_mask, best_rprj_view, best_points, best_projected, best_hypothesis


def loadTum(path):
    Rs, ts = [], []
    with open(path, 'r') as fp:
        for line in fp:
            splitted = line.rstrip().split(' ')
            t = [float(x) for x in splitted[1:4]]
            ts.append(t)
            q_xyzw = [float(x) for x in splitted[4:]]
            R = Rotation.from_quat(q_xyzw).as_matrix()
            Rs.append(R)
    return np.array(Rs), np.array(ts)


def loadIntrin(path):
    with open(path, 'r') as fp:
        for line in fp:
            splitted = line.rstrip().split(' ')
            w = int(splitted[0])
            h = int(splitted[1])
            fx = float(splitted[2])
            fy = float(splitted[3])
            cx = float(splitted[4])
            cy = float(splitted[5])
            return w, h, fx, fy, cx, cy


def loadLandmarks(path, lmk_num=68):
    lmk2ds = []
    valid_mask = []
    with open(path, 'r') as fp:
        for line in fp:
            splitted = line.rstrip().split(' ')
            if len(splitted) != lmk_num * 2:
                lmk2ds.append([[0, 0]] * lmk_num)
                valid_mask.append(False)
                continue
            lmk2d = []
            for i in range(lmk_num):
                lmk2d.append(
                    [float(splitted[2 * i]), float(splitted[2 * i + 1])])
            lmk2ds.append(lmk2d)
            valid_mask.append(True)
    return np.array(lmk2ds), np.array(valid_mask, dtype=bool)


if __name__ == '__main__':
    c2w_Rs, c2w_ts = loadTum("./data/tum.txt")
    R_gl2cv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    c2w_Rs = np.array([m @ R_gl2cv for m in c2w_Rs])
    w, h, fx, fy, cx, cy = loadIntrin("./data/intrin.txt")
    VIEW_NUM = c2w_Rs.shape[0]
    Ks = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                  )[None, ].repeat(VIEW_NUM, axis=0)
    w2c_Rs = c2w_Rs.transpose((0, 2, 1))
    w2c_ts = - np.matmul(w2c_Rs, c2w_ts[..., None]).squeeze(-1)

    lmk2ds, valid_mask = loadLandmarks("./data/detected/detected.txt")

    lmk2ds, w2c_Rs, w2c_ts, Ks = lmk2ds[valid_mask], w2c_Rs[valid_mask],\
        w2c_ts[valid_mask], Ks[valid_mask]

    os.makedirs("./data/triangulate", exist_ok=True)

    points3d = triangulateLandmarkLeastSquares(lmk2ds, w2c_Rs, w2c_ts, Ks)
    projected, rprj_err, rprj_err_view = evalReprojectionError(
        points3d, lmk2ds, w2c_Rs, w2c_ts, Ks)
    print('All views')
    print('Reprojection error: ', rprj_err_view)
    print()
    obj_io.saveObjSimple(
        "./data/triangulate/triangulated_all.obj", points3d, [])

    detected_img_names = [x for x in os.listdir(
        "./data/detected") if x.startswith('detected') and x.endswith('png')]

    for i, detected_img_name in enumerate(detected_img_names):
        img = cv2.imread("./data/detected/"+detected_img_name)
        for p in projected[i]:
            cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 255, 0))
        cv2.imwrite('./data/triangulate/rprj_all_' + detected_img_name, img)

    inlier_th = h * 0.02
    best_inlier_mask, best_rprj_view, best_points, best_projected, best_hypothesis =\
        triangulateLandmarkRANSAC(
            lmk2ds, w2c_Rs, w2c_ts, Ks, inlier_th)
    print('RANSAC inlier views', np.where(best_inlier_mask)[0])
    print('Hypothesis', best_hypothesis)
    print('Reprojection error: ', best_rprj_view)
    print()
    obj_io.saveObjSimple(
        "./data/triangulate/triangulated_ransac.obj", best_points, [])

    for i, detected_img_name in enumerate(detected_img_names):
        img = cv2.imread("./data/detected/"+detected_img_name)
        for p in best_projected[i]:
            cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 255, 0))
        cv2.imwrite('./data/triangulate/rprj_ransac_' + detected_img_name, img)
