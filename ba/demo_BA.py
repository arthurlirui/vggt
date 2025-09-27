import pyceres
from pyceres import LossFunction, Problem, Manifold
import pycolmap
import numpy as np
from hloc.utils import viz_3d

def create_reconstruction(num_points=50, num_images=2, seed=3, noise=0):
    state = np.random.RandomState(seed)
    rec = pycolmap.Reconstruction()
    p3d = state.uniform(-1, 1, (num_points, 3)) + np.array([0, 0, 3])
    for p in p3d:
        rec.add_point3D(p, pycolmap.Track(), np.zeros(3))
    w, h = 640, 480
    cam = pycolmap.Camera(
        model="SIMPLE_PINHOLE",
        width=w,
        height=h,
        params=np.array([max(w, h) * 1.2, w / 2, h / 2]),
        camera_id=0,
    )
    rec.add_camera(cam)
    for i in range(num_images):
        im = pycolmap.Image(
            id=i,
            name=str(i),
            camera_id=cam.camera_id,
            cam_from_world=pycolmap.Rigid3d(
                pycolmap.Rotation3d(), state.uniform(-1, 1, 3)
            ),
        )
        im.registered = True
        p2d = cam.img_from_cam(
            im.cam_from_world * [p.xyz for p in rec.points3D.values()]
        )
        p2d_obs = np.array(p2d) + state.randn(len(p2d), 2) * noise
        im.points2D = pycolmap.ListPoint2D(
            [pycolmap.Point2D(p, id_) for p, id_ in zip(p2d_obs, rec.points3D)]
        )
        rec.add_image(im)
    return rec

if __name__ == '__main__':
    rec_gt = create_reconstruction()