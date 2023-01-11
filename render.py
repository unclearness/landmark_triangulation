import pyrender
import numpy as np
import cv2
import trimesh
import math
from scipy.spatial.transform import Rotation


def renderMesh(trimesh_mesh, h, w, yfov, Rs, ts, unlit=False):
    if unlit:
        flags = pyrender.constants.RenderFlags.OFFSCREEN | pyrender.constants.RenderFlags.FLAT
    else:
        flags = pyrender.constants.RenderFlags.OFFSCREEN | pyrender.constants.RenderFlags.VERTEX_NORMALS
    mesh = pyrender.Mesh.from_trimesh(trimesh_mesh)
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0)
    renderer = pyrender.OffscreenRenderer(w, h)
    light = pyrender.PointLight([1, 1, 1], 1e9)
    colors = []
    depths = []
    for R, t in zip(Rs, ts):
        scene = pyrender.Scene()
        scene.add(mesh)
        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = t
        scene.add(camera, pose=c2w)
        if not unlit:
            scene.add(light, pose=c2w)
        color, depth = renderer.render(scene, flags=flags)
        colors.append(color)
        depths.append(depth)
    return colors, depths


if __name__ == '__main__':

    trimesh_mesh = trimesh.load("./data/max-planck.obj")

    radius = np.max(trimesh_mesh.bounds[1] - trimesh_mesh.bounds[0]) * 2
    center = trimesh_mesh.centroid
    Rs, ts = [], []

    base_R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    steps = 21
    for i in range(steps+1):
        theta = math.radians(180 * i / steps - 90)
        # print(theta)
        t = - radius * np.array([math.sin(theta), 0,
                                 math.cos(theta)]) + np.array(center)
        ts.append(t)
        theta = -theta
        R = np.array([[math.cos(theta), 0, -math.sin(theta)],
                      [0, 1, 0],
                      [math.sin(theta), 0, math.cos(theta)]]) @ base_R
        Rs.append(R)

    with open('./data/tum.txt', 'w') as fp:
        for i, (R, t) in enumerate(zip(Rs, ts)):
            q_xyzw = Rotation.from_matrix(R).as_quat()
            line = " ".join([str(i)] + [str(x) for x in t] + [str(x)
                                                              for x in q_xyzw]) + "\n"
            fp.write(line)

    fovy = math.radians(30)
    h, w = 512, 512
    colors, depths = renderMesh(trimesh_mesh, h, w,
                                fovy, Rs, ts, unlit=False)

    with open('./data/intrin.txt', 'w') as fp:
        fy = h * 0.5 / math.tan(fovy * 0.5)
        fx = fy
        cx = w / 2
        cy = h / 2
        fp.write(" ".join([str(w), str(h), str(fx),
                           str(fy), str(cx), str(cy)]) + "\n")

    for i in range(len(colors)):
        cv2.imwrite("./data/" + str(i).zfill(2) + ".png", colors[i])
