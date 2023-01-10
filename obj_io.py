
import dataclasses
import numpy as np


@dataclasses.dataclass
class ObjMesh:
    verts: np.array
    uvs: np.array = np.array([])
    normals: np.array = np.array([])
    indices: np.array = np.array([])
    uv_indices: np.array = np.array([])
    normal_indices: np.array = np.array([])
    vert_colors: np.array = np.array([])

    def eunsureNumpy(self):
        for k, v in self.__dict__.items():
            if not isinstance(getattr(self, k), np.ndarray):
                if 'indices' in k:
                    setattr(self, k, np.asarray(v, dtype=np.int32))
                else:
                    setattr(self, k, np.asarray(v, dtype=np.float))

    def ensureList(self):
        for k, v in self.__dict__.items():
            if isinstance(getattr(self, k), np.ndarray):
                setattr(self, k, v.tolist())


def loadObjSimple(obj_path):
    num_verts = 0
    num_uvs = 0
    num_normals = 0
    num_indices = 0
    verts = []
    uvs = []
    normals = []
    vert_colors = []
    indices = []
    uv_indices = []
    normal_indices = []
    for line in open(obj_path, "r"):
        vals = line.split()
        if len(vals) == 0:
            continue
        if vals[0] == "v":
            v = [float(x) for x in vals[1:4]]
            verts.append(v)
            if len(vals) == 7:
                vc = [float(x) for x in vals[4:7]]
                vert_colors.append(vc)
            num_verts += 1
        if vals[0] == "vt":
            vt = [float(x) for x in vals[1:3]]
            uvs.append(vt)
            num_uvs += 1
        if vals[0] == "vn":
            vn = [float(x) for x in vals[1:4]]
            normals.append(vn)
            num_normals += 1
        if vals[0] == "f":
            v_index = []
            uv_index = []
            n_index = []
            for f in vals[1:]:
                w = f.split("/")
                if num_verts > 0:
                    v_index.append(int(w[0]) - 1)
                if num_uvs > 0:
                    uv_index.append(int(w[1]) - 1)
                if num_normals > 0:
                    n_index.append(int(w[2]) - 1)
            indices.append(v_index)
            uv_indices.append(uv_index)
            normal_indices.append(n_index)
            num_indices += 1
    return verts, uvs, normals, indices, uv_indices,\
        normal_indices, vert_colors


def loadObj(obj_path, is_numpy=True):
    mesh = ObjMesh(*loadObjSimple(obj_path))
    if is_numpy:
        mesh.eunsureNumpy()
    else:
        mesh.ensureList()
    return mesh


def saveObjSimple(
    obj_path, verts, indices, uvs=[], normals=[], uv_indices=[],
        normal_indices=[], vert_colors=[], mat_file=None, mat_name=None):
    f_out = open(obj_path, "w")
    f_out.write("####\n")
    f_out.write("#\n")
    f_out.write("# verts: %s\n" % (len(verts)))
    f_out.write("# Faces: %s\n" % (len(indices)))
    f_out.write("#\n")
    f_out.write("####\n")
    if mat_file is not None:
        f_out.write("mtllib " + mat_file + "\n")
    for vi, v in enumerate(verts):
        vertstr = "v %s %s %s" % (v[0], v[1], v[2])
        if len(vert_colors) > 0:
            color = vert_colors[vi]
            vertstr += " %s %s %s" % (color[0], color[1], color[2])
        vertstr += "\n"
        f_out.write(vertstr)
    f_out.write("# %s verts\n\n" % (len(verts)))
    for uv in uvs:
        uvstr = "vt %s %s\n" % (uv[0], uv[1])
        f_out.write(uvstr)
    f_out.write("# %s uvs\n\n" % (len(uvs)))
    for n in normals:
        nStr = "vn %s %s %s\n" % (n[0], n[1], n[2])
        f_out.write(nStr)
    f_out.write("# %s normals\n\n" % (len(normals)))
    if mat_name is not None:
        f_out.write("usemtl " + mat_name + "\n")
    for fi, v_index in enumerate(indices):
        fStr = "f"
        for fvi, v_indexi in enumerate(v_index):
            fStr += " %s" % (v_indexi + 1)
            if len(uv_indices) > 0:
                fStr += "/%s" % (uv_indices[fi][fvi] + 1)
            if len(normal_indices) > 0:
                fStr += "/%s" % (normal_indices[fi][fvi] + 1)
        fStr += "\n"
        f_out.write(fStr)
    f_out.write("# %s faces\n\n" % (len(indices)))
    f_out.write("# End of File\n")
    f_out.close()


def saveObj(obj_path, mesh, mat_file=None, mat_name=None):
    saveObjSimple(obj_path, mesh.verts, mesh.indices, mesh.uvs,
                  mesh.normals, mesh.uv_indices,
                  mesh.normal_indices, mesh.vert_colors, mat_file, mat_name)
