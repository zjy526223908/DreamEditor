import os
import math
import cv2
import sys
from glob import glob
import json
import random
import numpy as np
import collections
from PIL import Image
from os import path
import torch
from torch.nn.functional import normalize
from torch.utils.data import DataLoader, Dataset
import open3d as o3d

from .provider_utils import auto_orient_and_center_poses, safe_normalize, radial_and_tangential_undistort, get_rays

import torch.nn.functional as F

DIR_COLORS = np.array([
    [255, 0, 0],  # front red
    [0, 255, 0],  # side  green
    [0, 0, 255],  # back  blue
    [255, 255, 0],  # side yellow
    [0, 0, 0],  # overhead
    [255, 0, 255],  # bottom
], dtype=np.uint8)


def visualize_poses(poses, size=0.02):
    # poses: [B, 4, 4], dirs: [B]

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    object = o3d.io.read_triangle_mesh('/mnt/c/Users/Administrator/PycharmProjects/'
                                       + 'stable-dreamfusion/data/blendedmvs/stone_horse/visual.ply')
    # object = o3d.io.read_point_cloud('/mnt/c/Users/Administrator/PycharmProjects/'
    #                                    +'stable-dreamfusion/data/blendedmvs/face/colmap.ply')
    objects = [axes, object]

    count = 0

    for pose in zip(poses):
        # a camera is visualized with 8 line segments.
        pose = pose[0]
        pos = pose[:3, 3]
        print(pos)

        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector([
            pos, a, b, c, d
        ])
        lineset.lines = o3d.utility.Vector2iVector([
            [0, 1], [0, 2], [0, 3], [0, 4],
            [1, 2], [2, 3], [3, 4], [4, 1],
        ])

        lineset.colors = o3d.utility.Vector3dVector(DIR_COLORS[[0]].repeat(8, 0))
        # o3d.io.write_line_set("./tmp_1/camera_{}.ply".format(count), lineset)
        count += 1
        objects.append(lineset)

    o3d.visualization.draw_geometries(objects, mesh_show_back_face=True)


def get_view_direction(thetas, phis, overhead, front):
    #                   phis [B,];          thetas: [B,]
    # front = 0         [0, front)
    # side (left) = 1   [front, 180)
    # back = 2          [180, 180+front)
    # side (right) = 3  [180+front, 360)
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    # first determine by phis
    res[(phis < front) & (phis > (2 * np.pi - front))] = 0
    res[(phis >= front) & (phis < (np.pi - front))] = 1
    res[(phis >= (np.pi - front)) & (phis < (np.pi + front))] = 2

    res[(phis >= (np.pi + front)) & (phis <= (2 * np.pi - front))] = 3
    # override by thetas

    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    return res


def rand_poses(size, device, radius_range=[1, 1.5], theta_range=[0, 180], phi_range=[0, 360], return_dirs=False,
               angle_overhead=30, angle_front=60, jitter=False, uniform_sphere_rate=0.5):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, pi]
        phi_range: [min, max], should be in [0, 2 * pi]
    Return:
        poses: [size, 4, 4]
    '''

    theta_range = np.deg2rad(theta_range)
    phi_range = np.deg2rad(phi_range)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)

    radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]

    if random.random() < uniform_sphere_rate:
        unit_centers = F.normalize(
            torch.stack([
                (torch.rand(size, device=device) - 0.5) * 2.0,
                torch.rand(size, device=device),
                (torch.rand(size, device=device) - 0.5) * 2.0,
            ], dim=-1), p=2, dim=1
        )
        thetas = torch.acos(unit_centers[:, 1])
        phis = torch.atan2(unit_centers[:, 0], unit_centers[:, 2])
        phis[phis < 0] += 2 * np.pi
        centers = unit_centers * radius.unsqueeze(-1)
    else:
        thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
        phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

        centers = torch.stack([
            radius * torch.sin(thetas) * torch.sin(phis),
            radius * torch.cos(thetas),
            radius * torch.sin(thetas) * torch.cos(phis),
        ], dim=-1)  # [B, 3]

    targets = 0

    # jitters
    if jitter:
        centers = centers + (torch.rand_like(centers) * 0.2 - 0.1)
        targets = targets + torch.randn_like(centers) * 0.2

    # lookat
    forward_vector = safe_normalize(targets - centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))

    if jitter:
        up_noise = torch.randn_like(up_vector) * 0.02
    else:
        up_noise = 0

    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1) + up_noise)

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((-right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)
    else:
        dirs = None

    return poses, dirs


def circle_poses(device, radius=1.25, theta=60, phi=0, angle_overhead=30, angle_front=60):
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)

    thetas = torch.FloatTensor([theta]).to(device)
    phis = torch.FloatTensor([phi]).to(device)

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1)  # [B, 3]

    # lookat
    forward_vector = - safe_normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float).unsqueeze(0).to(device)

    poses[:, :3, :3] = torch.stack((-right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)

    return poses, dirs


Rays = collections.namedtuple(
    'Rays',
    ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far'))
Rays_keys = Rays._fields


def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*map(fn, tup))


class BaseDataset(Dataset):
    """BaseDataset Base Class."""

    def __init__(self, data_dir, split, if_data_cuda=True, batch_type='all_images', factor=0):
        super(BaseDataset, self).__init__()
        self.near = 2
        self.far = 6
        self.split = split
        self.data_dir = data_dir

        self.batch_type = batch_type
        self.images = None
        self.rays = None
        self.if_data_cuda = if_data_cuda
        self.it = -1
        self.n_examples = 1
        self.factor = factor

    def _flatten(self, x):
        # Always flatten out the height x width dimensions
        x = [y.reshape([-1, y.shape[-1]]) for y in x]
        if self.batch_type == 'all_images':
            # If global batching, also concatenate all data into one list
            out_size = [len(x)] + list(x[0].shape)
            x = np.concatenate(x, axis=0).reshape(out_size)
            if self.if_data_cuda:
                x = torch.tensor(x).cuda()
            else:
                x = torch.tensor(x)
        else:
            if self.if_data_cuda:
                x = [torch.tensor(y).cuda() for y in x]
            else:
                x = [torch.tensor(y) for y in x]
        return x

    def _train_init(self):
        """Initialize training."""

        self._load_renderings()
        self._generate_rays()

        if self.split == 'train':
            self.images = self._flatten(self.images)
            self.masks = self._flatten(self.masks)
            self.origins = self._flatten(self.origins)
            self.directions = self._flatten(self.directions)

            # self.rays = namedtuple_map(self._flatten, self.rays)

    def _val_init(self):
        self._load_renderings()
        self._generate_rays()

        self.images = self._flatten(self.images)
        self.masks = self._flatten(self.masks)
        self.origins = self._flatten(self.origins)
        self.directions = self._flatten(self.directions)

    def _generate_rays(self):
        """Generating rays for all images."""
        raise ValueError('Implement in different dataset.')

    def _load_renderings(self):
        raise ValueError('Implement in different dataset.')

    def __len__(self):

        return self.n_images

    def __getitem__(self, index):
        if self.split == 'val':
            index = (self.it + 1) % self.n_examples
            self.it += 1
        # rays = Rays(*[getattr(self.rays, key)[index] for key in Rays_keys])
        # index = torch.tensor(index)

        return self.images[index], self.masks[index], self.origins[index], self.directions[index], self.H[index], \
        self.W[index]


class InstantNGPData(BaseDataset):
    """Blender Dataset."""

    def __init__(self, data_dir, if_data_cuda=True, if_sphere=False, R_path=None,
                 resolution_level=1, split='train', batch_type='all_images', factor=0, opt=None):
        super(InstantNGPData, self).__init__(data_dir, split, if_data_cuda, batch_type, factor)
        self.resolution_level = resolution_level
        self.near = 0.01
        self.far = 0.5
        self.if_sphere = if_sphere
        self.if_data_cuda = if_data_cuda
        self.R_path = R_path
        self.scene_scale = opt.scene_scale

        print(self.if_data_cuda, batch_type)

        self._train_init()

    def _load_renderings(self):
        """Load images from disk."""

        with open(os.path.join(self.data_dir, "transforms.json"), encoding="UTF-8") as file:
            self.meta = json.load(file)

        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.*')))
        self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.*')))

        self.n_images = len(self.images_lis)
        # self.n_images = 80

        poses = []

        for frame in self.meta["frames"]:
            poses.append(np.array(frame["transform_matrix"]))
        poses = np.array(poses).astype(np.float32)
        poses[:, :3, 3] *= self.scene_scale
        self.camera_to_world = torch.from_numpy(poses[:, :3])

        if self.R_path:
            self.pose_optimizer = torch.tensor(np.load(self.R_path))
        else:
            self.pose_optimizer = torch.eye(4).unsqueeze(0).repeat(self.n_images, 1, 1)

        images = []
        H = []
        W = []
        for i, image_path in enumerate(self.images_lis):
            image = np.array(Image.open(image_path), dtype=np.float32) / 256.
            image = cv2.resize(image, (
            int(image.shape[1] / self.resolution_level), int(image.shape[0] / self.resolution_level)),
                               interpolation=cv2.INTER_AREA)
            H.append(int(image.shape[0]))
            W.append(int(image.shape[1]))
            images.append(image)

        masks = []
        for i, mask_path in enumerate(self.masks_lis):
            mask = np.array(Image.open(mask_path), dtype=np.float32) / 256.
            mask = cv2.resize(mask,
                              (int(mask.shape[1] / self.resolution_level), int(mask.shape[0] / self.resolution_level)),
                              interpolation=cv2.INTER_AREA)
            if len(mask.shape) == 3:
                mask = np.sum(mask, axis=2)
            mask[mask > 0] = True
            masks.append(mask)

        self.images = images
        self.masks = masks
        print(self.images[0].shape)
        self.H, self.W = H, W

    def get_focal_lengths(self, meta):
        """Reads or computes the focal length from transforms dict.
        Args:
            meta: metadata from transforms.json file.
        Returns:
            Focal lengths in the x and y directions. Error is raised if these cannot be calculated.
        """
        fl_x, fl_y = 0, 0

        def fov_to_focal_length(rad, res):
            return 0.5 * res / np.tan(0.5 * rad)

        if "fl_x" in meta:
            fl_x = meta["fl_x"]
        elif "x_fov" in meta:
            fl_x = fov_to_focal_length(np.deg2rad(meta["x_fov"]), meta["w"])
        elif "camera_angle_x" in meta:
            fl_x = fov_to_focal_length(meta["camera_angle_x"], meta["w"])

        if "fl_y" in meta:
            fl_y = meta["fl_y"]
        elif "y_fov" in meta:
            fl_y = fov_to_focal_length(np.deg2rad(meta["y_fov"]), meta["h"])
        elif "camera_angle_y" in meta:
            fl_y = fov_to_focal_length(meta["camera_angle_y"], meta["h"])

        if fl_x == 0 or fl_y == 0:
            raise AttributeError("Focal length cannot be calculated from transforms.json (missing fields).")

        return (fl_x, fl_y)

    def multiply(self, pose_a, pose_b):
        """Multiply two pose matrices, A @ B.

        Args:
            pose_a: Left pose matrix, usually a transformation applied to the right.
            pose_b: Right pose matrix, usually a camera pose that will be tranformed by pose_a.

        Returns:
            Camera pose matrix where pose_a was applied to pose_b.
        """
        R1, t1 = pose_a[..., :3, :3], pose_a[..., :3, 3:]
        R2, t2 = pose_b[..., :3, :3], pose_b[..., :3, 3:]
        R = R1.matmul(R2)
        t = t1 + R1.matmul(t2)
        return torch.cat([R, t], dim=-1)

    def _generate_rays(self):
        """Generating rays for all images."""

        fx, fy = self.get_focal_lengths(self.meta)

        cx = float(self.meta["cx"]),
        cy = float(self.meta["cy"]),
        height = int(self.meta["h"]),
        width = int(self.meta["w"]),
        cx = cx[0]
        cy = cy[0]

        origins_list = []
        directions_list = []

        for i in range(self.n_images):
            W = self.images[i].shape[1]
            H = self.images[i].shape[0]

            l = self.resolution_level
            tx = torch.linspace(0, W * l - 1, W)
            ty = torch.linspace(0, H * l - 1, H)

            x, y = torch.meshgrid(tx, ty)
            x = x + 0.5
            y = y + 0.5

            x = x.reshape(-1)
            y = y.reshape(-1)

            coord = torch.stack([(x - cx) / fx, -(y - cy) / fy], -1)  # (num_rays, 2)
            coord_x_offset = torch.stack([(x - cx + 1) / fx, -(y - cy) / fy], -1)  # (num_rays, 2)
            coord_y_offset = torch.stack([(x - cx) / fx, -(y - cy + 1) / fy], -1)  # (num_rays, 2)

            coord_stack = torch.stack([coord, coord_x_offset, coord_y_offset], dim=0)  # (3, num_rays, 2)

            directions_stack = torch.empty((3,) + (coord_stack.shape[1],) + (3,))

            directions_stack[..., 0] = coord_stack[..., 0].float()
            directions_stack[..., 1] = coord_stack[..., 1].float()
            directions_stack[..., 2] = -1.0

            c2w = self.camera_to_world[i].unsqueeze(0)  # .repeat(coord_stack.shape[1], 1, 1)
            camera_opt_to_camera = self.pose_optimizer[i].unsqueeze(0)

            c2w = self.multiply(c2w, camera_opt_to_camera)
            c2w = c2w.repeat(coord_stack.shape[1], 1, 1)

            rotation = c2w[..., :3, :3]

            directions_stack = torch.sum(
                directions_stack[..., None, :] * rotation, dim=-1
            )

            directions_norm = torch.norm(directions_stack, dim=-1, keepdim=True)
            directions_norm = directions_norm[0]

            directions_stack = normalize(directions_stack, dim=-1)

            origins = c2w[..., :3, 3]
            directions = directions_stack[0]

            origins = origins.reshape(W, H, 3)
            directions = directions.reshape(W, H, 3)

            origins = origins.permute(1, 0, 2)
            directions = directions.permute(1, 0, 2)

            origins_list.append(origins.numpy())
            directions_list.append(directions.numpy())

        self.origins = origins_list
        self.directions = directions_list


class NerfstudioRayData(BaseDataset):
    """Blender Dataset."""

    def __init__(self, data_dir, if_data_cuda=True, if_sphere=False, R_path=None,
                 resolution_level=1, split='train', batch_type='all_images', factor=0):
        super(NerfstudioRayData, self).__init__(data_dir, split, if_data_cuda, batch_type, factor)
        self.resolution_level = resolution_level
        self.near = 0.01
        self.far = 0.5
        self.if_sphere = if_sphere
        self.if_data_cuda = if_data_cuda
        self.R_path = R_path
        self.scene_scale = 0.33
        print(self.if_data_cuda, batch_type)

        self._train_init()

    def _load_renderings(self):
        """Load images from disk."""

        with open(os.path.join(self.data_dir, "transforms.json"), encoding="UTF-8") as file:
            self.meta = json.load(file)

        poses = []

        for frame in self.meta["frames"]:
            poses.append(np.array(frame["transform_matrix"]))
        poses = np.array(poses).astype(np.float32)

        poses, transform_matrix = auto_orient_and_center_poses(
            torch.tensor(poses),
            method="up",
            center_poses=True,
        )
        poses = poses.numpy()
        scale_factor = 1.0
        scale_factor /= float(torch.max(torch.abs(torch.tensor(poses[:, :3, 3]))))
        poses[:, :3, 3] *= scale_factor

        self.n_images = poses.shape[0]

        self.if_distortion = True if self.meta["camera_model"] == 'OPENCV_FISHEYE' else False
        self.camera_to_world = torch.from_numpy(poses[:, :3])

        if self.R_path:
            self.pose_optimizer = torch.tensor(np.load(self.R_path))
        else:
            self.pose_optimizer = torch.eye(4).unsqueeze(0).repeat(self.n_images, 1, 1)

        images = []
        H = []
        W = []

        print(self.n_images)
        self.H, self.W = H, W

    def get_focal_lengths(self, meta):
        """Reads or computes the focal length from transforms dict.
        Args:
            meta: metadata from transforms.json file.
        Returns:
            Focal lengths in the x and y directions. Error is raised if these cannot be calculated.
        """
        fl_x, fl_y = 0, 0

        def fov_to_focal_length(rad, res):
            return 0.5 * res / np.tan(0.5 * rad)

        if "fl_x" in meta:
            fl_x = meta["fl_x"]
        elif "x_fov" in meta:
            fl_x = fov_to_focal_length(np.deg2rad(meta["x_fov"]), meta["w"])
        elif "camera_angle_x" in meta:
            fl_x = fov_to_focal_length(meta["camera_angle_x"], meta["w"])

        if "fl_y" in meta:
            fl_y = meta["fl_y"]
        elif "y_fov" in meta:
            fl_y = fov_to_focal_length(np.deg2rad(meta["y_fov"]), meta["h"])
        elif "camera_angle_y" in meta:
            fl_y = fov_to_focal_length(meta["camera_angle_y"], meta["h"])

        if fl_x == 0 or fl_y == 0:
            raise AttributeError("Focal length cannot be calculated from transforms.json (missing fields).")

        return (fl_x, fl_y)

    def multiply(self, pose_a, pose_b):
        """Multiply two pose matrices, A @ B.

        Args:
            pose_a: Left pose matrix, usually a transformation applied to the right.
            pose_b: Right pose matrix, usually a camera pose that will be tranformed by pose_a.

        Returns:
            Camera pose matrix where pose_a was applied to pose_b.
        """
        R1, t1 = pose_a[..., :3, :3], pose_a[..., :3, 3:]
        R2, t2 = pose_b[..., :3, :3], pose_b[..., :3, 3:]
        R = R1.matmul(R2)
        t = t1 + R1.matmul(t2)
        return torch.cat([R, t], dim=-1)

    def _generate_rays(self):
        """Generating rays for all images."""

        cx = float(self.meta["cx"]),
        cy = float(self.meta["cy"]),
        fx = float(self.meta["fl_x"])
        fy = float(self.meta["fl_y"])
        height = int(self.meta["h"]),
        width = int(self.meta["w"]),
        k1 = float(self.meta["k1"]) if "k1" in self.meta else 0.0,
        k2 = float(self.meta["k2"]) if "k2" in self.meta else 0.0,
        k3 = float(self.meta["k3"]) if "k3" in self.meta else 0.0,
        k4 = float(self.meta["k4"]) if "k4" in self.meta else 0.0,
        p1 = float(self.meta["p1"]) if "p1" in self.meta else 0.0,
        p2 = float(self.meta["p2"]) if "p2" in self.meta else 0.0,
        distortion_params = torch.Tensor([k1, k2, k3, k4, p1, p2])

        cx = cx[0]
        cy = cy[0]

        origins_list = []
        directions_list = []

        for i in range(self.n_images):

            l = self.resolution_level

            W = width[0] // l
            H = height[0] // l

            tx = torch.linspace(0, W * l - 1, W)
            ty = torch.linspace(0, H * l - 1, H)

            x, y = torch.meshgrid(tx, ty)
            x = x + 0.5
            y = y + 0.5

            x = x.reshape(-1)
            y = y.reshape(-1)

            coord = torch.stack([(x - cx) / fx, -(y - cy) / fy], -1)  # (num_rays, 2)
            coord_x_offset = torch.stack([(x - cx + 1) / fx, -(y - cy) / fy], -1)  # (num_rays, 2)
            coord_y_offset = torch.stack([(x - cx) / fx, -(y - cy + 1) / fy], -1)  # (num_rays, 2)

            coord_stack = torch.stack([coord, coord_x_offset, coord_y_offset], dim=0)  # (3, num_rays, 2)
            directions_stack = torch.empty((3,) + (coord_stack.shape[1],) + (3,))
            if self.if_distortion:
                coord_stack = radial_and_tangential_undistort(
                    coord_stack,
                    distortion_params.squeeze().unsqueeze(0).repeat(coord_stack.shape[1], 1),
                )

                theta = torch.sqrt(torch.sum(coord_stack ** 2, dim=-1))
                theta = torch.clip(theta, 0.0, math.pi)
                sin_theta = torch.sin(theta)

                directions_stack[..., 0] = coord_stack[..., 0] * sin_theta / theta
                directions_stack[..., 1] = coord_stack[..., 1] * sin_theta / theta
                directions_stack[..., 2] = -torch.cos(theta)

            else:
                directions_stack[..., 0] = coord_stack[..., 0].float()
                directions_stack[..., 1] = coord_stack[..., 1].float()
                directions_stack[..., 2] = -1.0

            c2w = self.camera_to_world[i].unsqueeze(0)  # .repeat(coord_stack.shape[1], 1, 1)
            if self.R_path:
                camera_opt_to_camera = self.pose_optimizer[i].unsqueeze(0)
                c2w = self.multiply(c2w, camera_opt_to_camera)
            c2w = c2w.repeat(coord_stack.shape[1], 1, 1)

            rotation = c2w[..., :3, :3]

            directions_stack = torch.sum(
                directions_stack[..., None, :] * rotation, dim=-1
            )

            directions_norm = torch.norm(directions_stack, dim=-1, keepdim=True)
            directions_norm = directions_norm[0]

            directions_stack = normalize(directions_stack, dim=-1)

            origins = c2w[..., :3, 3]
            directions = directions_stack[0]

            origins = origins.reshape(W, H, 3)
            directions = directions.reshape(W, H, 3)

            origins = origins.permute(1, 0, 2)
            directions = directions.permute(1, 0, 2)

            origins_list.append(origins.numpy())
            directions_list.append(directions.numpy())

        self.origins = origins_list
        self.directions = directions_list

    def _train_init(self):
        """Initialize training."""

        self._load_renderings()
        self._generate_rays()

        if self.split == 'train':
            self.origins = self._flatten(self.origins)
            self.directions = self._flatten(self.directions)

    def __len__(self):

        return self.n_images

    def __getitem__(self, index):
        if self.split == 'val':
            index = (self.it + 1) % self.n_examples
            self.it += 1
        # rays = Rays(*[getattr(self.rays, key)[index] for key in Rays_keys])
        # index = torch.tensor(index)

        return self.origins[index], self.directions[index]


class NerfstudioData(BaseDataset):
    """Blender Dataset."""

    def __init__(self, data_dir, if_data_cuda=True, if_sphere=False, R_path=None,
                 resolution_level=1, split='train', batch_type='all_images', factor=0):
        super(NerfstudioData, self).__init__(data_dir, split, if_data_cuda, batch_type, factor)
        self.resolution_level = resolution_level
        self.near = 0.01
        self.far = 0.5
        self.if_sphere = if_sphere
        self.if_data_cuda = if_data_cuda
        self.R_path = R_path
        self.scene_scale = 0.33
        print(self.if_data_cuda, batch_type)

        self._train_init()

    def _load_renderings(self):
        """Load images from disk."""

        with open(os.path.join(self.data_dir, "transforms.json"), encoding="UTF-8") as file:
            self.meta = json.load(file)

        self.images_lis = []
        self.masks_lis = []

        poses = []

        for frame in self.meta["frames"]:
            poses.append(np.array(frame["transform_matrix"]))
            self.images_lis.append(os.path.join(self.data_dir, frame["file_path"]))
            self.masks_lis.append(os.path.join(self.data_dir, frame["file_path"]))
        poses = np.array(poses).astype(np.float32)

        poses, transform_matrix = auto_orient_and_center_poses(
            torch.tensor(poses),
            method="up",
            center_poses=True,
        )
        poses = poses.numpy()
        scale_factor = 1.0
        scale_factor /= float(torch.max(torch.abs(torch.tensor(poses[:, :3, 3]))))
        poses[:, :3, 3] *= scale_factor

        # self.images_lis = self.images_lis[:10]
        # self.masks_lis = self.masks_lis[:10]

        num_images = len(self.images_lis)
        num_train_images = math.ceil(num_images * 0.9)

        i_train = np.linspace(
            0, num_images - 1, num_train_images, dtype=int
        )

        self.images_lis = [self.images_lis[i] for i in i_train]
        self.masks_lis = [self.masks_lis[i] for i in i_train]
        poses = poses[i_train]

        self.n_images = len(self.images_lis)

        # self.n_images = 80

        self.if_distortion = True if self.meta["camera_model"] == 'OPENCV_FISHEYE' else False
        self.camera_to_world = torch.from_numpy(poses[:, :3])
        # self.camera_to_world = self.camera_to_world[:10]

        if self.R_path:
            self.pose_optimizer = torch.tensor(np.load(self.R_path))
        else:
            self.pose_optimizer = torch.eye(4).unsqueeze(0).repeat(self.n_images, 1, 1)

        images = []
        H = []
        W = []
        for i, image_path in enumerate(self.images_lis):
            image = np.array(Image.open(image_path), dtype=np.float32) / 256.
            image = cv2.resize(image, (
            int(image.shape[1] / self.resolution_level), int(image.shape[0] / self.resolution_level)),
                               interpolation=cv2.INTER_AREA)
            H.append(int(image.shape[0]))
            W.append(int(image.shape[1]))
            images.append(image)

        masks = []
        for i, mask_path in enumerate(self.masks_lis):
            mask = np.array(Image.open(mask_path), dtype=np.float32) / 256.
            mask = cv2.resize(mask,
                              (int(mask.shape[1] / self.resolution_level), int(mask.shape[0] / self.resolution_level)),
                              interpolation=cv2.INTER_AREA)
            if len(mask.shape) == 3:
                mask = np.sum(mask, axis=2)
            mask[mask > 0] = True
            masks.append(mask)

        self.images = images
        self.masks = masks
        print(self.images[0].shape)
        self.H, self.W = H, W

    def get_focal_lengths(self, meta):
        """Reads or computes the focal length from transforms dict.
        Args:
            meta: metadata from transforms.json file.
        Returns:
            Focal lengths in the x and y directions. Error is raised if these cannot be calculated.
        """
        fl_x, fl_y = 0, 0

        def fov_to_focal_length(rad, res):
            return 0.5 * res / np.tan(0.5 * rad)

        if "fl_x" in meta:
            fl_x = meta["fl_x"]
        elif "x_fov" in meta:
            fl_x = fov_to_focal_length(np.deg2rad(meta["x_fov"]), meta["w"])
        elif "camera_angle_x" in meta:
            fl_x = fov_to_focal_length(meta["camera_angle_x"], meta["w"])

        if "fl_y" in meta:
            fl_y = meta["fl_y"]
        elif "y_fov" in meta:
            fl_y = fov_to_focal_length(np.deg2rad(meta["y_fov"]), meta["h"])
        elif "camera_angle_y" in meta:
            fl_y = fov_to_focal_length(meta["camera_angle_y"], meta["h"])

        if fl_x == 0 or fl_y == 0:
            raise AttributeError("Focal length cannot be calculated from transforms.json (missing fields).")

        return (fl_x, fl_y)

    def multiply(self, pose_a, pose_b):
        """Multiply two pose matrices, A @ B.

        Args:
            pose_a: Left pose matrix, usually a transformation applied to the right.
            pose_b: Right pose matrix, usually a camera pose that will be tranformed by pose_a.

        Returns:
            Camera pose matrix where pose_a was applied to pose_b.
        """
        R1, t1 = pose_a[..., :3, :3], pose_a[..., :3, 3:]
        R2, t2 = pose_b[..., :3, :3], pose_b[..., :3, 3:]
        R = R1.matmul(R2)
        t = t1 + R1.matmul(t2)
        return torch.cat([R, t], dim=-1)

    def _generate_rays(self):
        """Generating rays for all images."""

        cx = float(self.meta["cx"]),
        cy = float(self.meta["cy"]),
        fx = float(self.meta["fl_x"])
        fy = float(self.meta["fl_y"])
        height = int(self.meta["h"]),
        width = int(self.meta["w"]),
        k1 = float(self.meta["k1"]) if "k1" in self.meta else 0.0,
        k2 = float(self.meta["k2"]) if "k2" in self.meta else 0.0,
        k3 = float(self.meta["k3"]) if "k3" in self.meta else 0.0,
        k4 = float(self.meta["k4"]) if "k4" in self.meta else 0.0,
        p1 = float(self.meta["p1"]) if "p1" in self.meta else 0.0,
        p2 = float(self.meta["p2"]) if "p2" in self.meta else 0.0,
        distortion_params = torch.Tensor([k1, k2, k3, k4, p1, p2])

        cx = cx[0]
        cy = cy[0]

        origins_list = []
        directions_list = []

        for i in range(self.n_images):

            W = self.images[i].shape[1]
            H = self.images[i].shape[0]

            l = self.resolution_level
            tx = torch.linspace(0, W * l - 1, W)
            ty = torch.linspace(0, H * l - 1, H)

            x, y = torch.meshgrid(tx, ty)
            x = x + 0.5
            y = y + 0.5

            x = x.reshape(-1)
            y = y.reshape(-1)

            coord = torch.stack([(x - cx) / fx, -(y - cy) / fy], -1)  # (num_rays, 2)
            coord_x_offset = torch.stack([(x - cx + 1) / fx, -(y - cy) / fy], -1)  # (num_rays, 2)
            coord_y_offset = torch.stack([(x - cx) / fx, -(y - cy + 1) / fy], -1)  # (num_rays, 2)

            coord_stack = torch.stack([coord, coord_x_offset, coord_y_offset], dim=0)  # (3, num_rays, 2)
            directions_stack = torch.empty((3,) + (coord_stack.shape[1],) + (3,))
            if self.if_distortion:
                coord_stack = radial_and_tangential_undistort(
                    coord_stack,
                    distortion_params.squeeze().unsqueeze(0).repeat(coord_stack.shape[1], 1),
                )

                theta = torch.sqrt(torch.sum(coord_stack ** 2, dim=-1))
                theta = torch.clip(theta, 0.0, math.pi)
                sin_theta = torch.sin(theta)

                directions_stack[..., 0] = coord_stack[..., 0] * sin_theta / theta
                directions_stack[..., 1] = coord_stack[..., 1] * sin_theta / theta
                directions_stack[..., 2] = -torch.cos(theta)

            else:
                directions_stack[..., 0] = coord_stack[..., 0].float()
                directions_stack[..., 1] = coord_stack[..., 1].float()
                directions_stack[..., 2] = -1.0

            c2w = self.camera_to_world[i].unsqueeze(0)  # .repeat(coord_stack.shape[1], 1, 1)
            if self.R_path:
                camera_opt_to_camera = self.pose_optimizer[i].unsqueeze(0)
                c2w = self.multiply(c2w, camera_opt_to_camera)
            c2w = c2w.repeat(coord_stack.shape[1], 1, 1)

            rotation = c2w[..., :3, :3]

            directions_stack = torch.sum(
                directions_stack[..., None, :] * rotation, dim=-1
            )

            directions_norm = torch.norm(directions_stack, dim=-1, keepdim=True)
            directions_norm = directions_norm[0]

            directions_stack = normalize(directions_stack, dim=-1)

            origins = c2w[..., :3, 3]
            directions = directions_stack[0]

            origins = origins.reshape(W, H, 3)
            directions = directions.reshape(W, H, 3)

            origins = origins.permute(1, 0, 2)
            directions = directions.permute(1, 0, 2)

            origins_list.append(origins.numpy())
            directions_list.append(directions.numpy())

        self.origins = origins_list
        self.directions = directions_list


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


class DTU(BaseDataset):
    """Blender Dataset."""

    def __init__(self, data_dir, if_data_cuda=True, if_sphere=False, R_path=None,
                 resolution_level=1, split='train', batch_type='all_images', factor=0):
        super(DTU, self).__init__(data_dir, split, if_data_cuda, batch_type, factor)
        self.resolution_level = resolution_level
        self.near = 0.01
        self.far = 0.5
        self.if_sphere = if_sphere
        self.if_data_cuda = if_data_cuda
        self.R_path = R_path
        print(self.if_data_cuda, batch_type)

        self._train_init()

    def _load_renderings(self):
        """Load images from disk."""

        if self.if_sphere:
            print('use cameras_sphere.npz')
            camera_dict = np.load(path.join(self.data_dir, 'cameras_sphere.npz'))
        else:
            print('use cameras_large.npz')
            camera_dict = np.load(path.join(self.data_dir, 'cameras_large.npz'))
        self.camera_dict = camera_dict

        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))
        self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))

        self.n_images = len(self.images_lis)
        # self.n_images = 80

        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]

            intrinsics, pose = load_K_Rt_from_P(None, P)

            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.intrinsics_all = torch.stack(self.intrinsics_all)  # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.pose_all = torch.stack(self.pose_all)

        if self.R_path:
            R = np.load(self.R_path)
            R = torch.tensor(R).float()
            self.pose_all = R @ self.pose_all

        # visualize_poses(self.pose_all.numpy())

        images = []
        H = []
        W = []
        for i, image_path in enumerate(self.images_lis):
            image = np.array(Image.open(image_path), dtype=np.float32) / 256.
            image = cv2.resize(image, (
            int(image.shape[1] / self.resolution_level), int(image.shape[0] / self.resolution_level)),
                               interpolation=cv2.INTER_AREA)
            H.append(int(image.shape[0]))
            W.append(int(image.shape[1]))
            images.append(image)

        masks = []
        for i, mask_path in enumerate(self.masks_lis):
            mask = np.array(Image.open(mask_path), dtype=np.float32) / 256.
            mask = cv2.resize(mask,
                              (int(mask.shape[1] / self.resolution_level), int(mask.shape[0] / self.resolution_level)),
                              interpolation=cv2.INTER_AREA)
            if len(mask.shape) == 3:
                mask = np.sum(mask, axis=2)
            mask[mask > 0] = True
            masks.append(mask)

        self.images = images
        self.masks = masks
        print(self.images[0].shape)
        self.H, self.W = H, W

    def _generate_rays(self):
        """Generating rays for all images."""

        origins = []
        directions = []

        for i in range(self.n_images):
            W = self.images[i].shape[1]
            H = self.images[i].shape[0]

            l = self.resolution_level
            tx = torch.linspace(0, W * l - 1, W)
            ty = torch.linspace(0, H * l - 1, H)

            pixels_x, pixels_y = torch.meshgrid(tx, ty)
            p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
            p = torch.matmul(self.intrinsics_all_inv[i, None, None, :3, :3],
                             p[:, :, :, None]).squeeze()  # W, H, 3
            rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
            rays_v = torch.matmul(self.pose_all[i, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3

            rays_v = rays_v / torch.linalg.norm(rays_v, ord=2, dim=-1, keepdim=True)  # W, H, 3

            rays_o = self.pose_all[i, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3

            rays_o = rays_o.permute(1, 0, 2)
            rays_v = rays_v.permute(1, 0, 2)

            origins.append(rays_o.numpy())
            directions.append(rays_v.numpy())

        viewdirs = [
            v / np.linalg.norm(v, axis=-1, keepdims=True) for v in directions
        ]

        def broadcast_scalar_attribute(x):
            return [
                x * np.ones_like(origins[i][..., :1])
                for i in range(self.n_images)
            ]

        lossmults = broadcast_scalar_attribute(1).copy()
        nears = broadcast_scalar_attribute(self.near).copy()
        fars = broadcast_scalar_attribute(self.far).copy()

        # Distance from each unit-norm direction vector to its x-axis neighbor.
        dx = [
            np.sqrt(np.sum((v[:-1, :, :] - v[1:, :, :]) ** 2, -1)) for v in directions
        ]
        dx = [np.concatenate([v, v[-2:-1, :]], 0) for v in dx]
        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.

        radii = [v[..., None] * 2 / np.sqrt(12) for v in dx]

        self.origins = origins
        self.directions = directions

        del origins, directions, viewdirs, radii, lossmults, nears, fars


class NeRFDataset:
    def __init__(self, opt, device, R_path=None, type='train', H=256, W=256, size=100):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type  # train, val, test

        self.H = H
        self.W = W
        self.radius_range = opt.radius_range
        self.fovy_range = opt.fovy_range
        self.size = size

        self.training = self.type in ['train', 'all']

        if self.training:
            resolution_level = self.opt.train_resolution_level
        else:
            resolution_level = self.opt.eval_resolution_level

        if opt.data_type == 'dtu':
            self.dataset = DTU(data_dir=self.opt.data_path, if_data_cuda=self.opt.if_data_cuda,
                               if_sphere=self.opt.if_sphere,
                               R_path=R_path, resolution_level=resolution_level, split='train',
                               batch_type=self.opt.train_batch_type, factor=1)
        elif opt.data_type == 'nerfstudio':
            self.dataset = NerfstudioData(data_dir=self.opt.data_path, if_data_cuda=self.opt.if_data_cuda,
                                          if_sphere=self.opt.if_sphere,
                                          R_path=R_path, resolution_level=resolution_level, split='train',
                                          batch_type=self.opt.train_batch_type, factor=1)
        elif opt.data_type == 'instantngp':
            self.dataset = InstantNGPData(data_dir=self.opt.data_path, if_data_cuda=self.opt.if_data_cuda,
                                          if_sphere=self.opt.if_sphere,
                                          R_path=R_path, resolution_level=resolution_level, split='train',
                                          batch_type=self.opt.train_batch_type,
                                          factor=1, opt=opt)
        else:
            print('unsupport data type')
            sys.exit()

    def dataloader(self):
        if self.training:
            loader = DataLoader(self.dataset, shuffle=True, num_workers=self.opt.num_work,
                                batch_size=self.opt.batch_size, pin_memory=False)
        else:
            loader = DataLoader(self.dataset, shuffle=False, num_workers=0,
                                batch_size=1, pin_memory=False)

        loader._data = self  # an ugly fix... we need to access dataset in trainer.
        return loader


class SphericalSamplingDataset:
    def __init__(self, opt, device, R_path=None, type='train', H=256, W=256, size=100):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type  # train, val, test

        self.H = H
        self.W = W
        self.radius_range = opt.radius_range
        self.fovy_range = opt.fovy_range
        self.phi_range = opt.phi_range
        self.theta_range = opt.theta_range
        self.size = size

        self.training = self.type in ['train', 'all']

        self.cx = self.H / 2
        self.cy = self.W / 2

        self.R_path = R_path

    def collate(self, index):

        if self.training:
            phi = random.uniform(self.phi_range[0], self.phi_range[1])
            theta = random.uniform(self.theta_range[0], self.theta_range[1])
            radius = random.uniform(self.radius_range[0], self.radius_range[1])

            poses, dirs = circle_poses(self.device, radius=radius, theta=theta, phi=phi,
                                       angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front)

            # random focal
            fov = random.random() * (self.fovy_range[1] - self.fovy_range[0]) + self.fovy_range[0]
            focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
            intrinsics = np.array([focal, focal, self.cy, self.cx])
        else:
            # circle pose
            phi = self.phi_range[0] + (index[0] / self.size) * (self.phi_range[1] - self.phi_range[0])
            theta = 0.5 * (self.theta_range[0] + self.theta_range[1])

            poses, dirs = circle_poses(self.device, radius=self.radius_range[0], theta=theta, phi=phi,
                                       angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front)

            # fixed focal
            fov = (self.fovy_range[1] + self.fovy_range[0]) / 2
            focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
            intrinsics = np.array([focal, focal, self.cy, self.cx])

        if self.R_path:
            R = np.load(self.R_path)
            R = torch.tensor(R).to(self.device).float()
            R = torch.inverse(R)
            poses = R @ poses

        rays = get_rays(poses, intrinsics, self.H, self.W, -1)

        data = {
            'H': self.H,
            'W': self.W,
            'phi': phi,
            'theta': theta,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'dir': dirs,
        }

        return data

    def dataloader(self):
        loader = DataLoader(list(range(self.size)), batch_size=1, collate_fn=self.collate, shuffle=self.training,
                            num_workers=0)
        return loader


class SampleViewsDataset:
    def __init__(self, opt, device, R_path=None, type='train', H=256, W=256, size=100):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type  # train, val, test

        self.H = H
        self.W = W
        self.radius_range = opt.radius_range
        self.fovy_range = opt.fovy_range
        self.phi_list = opt.phi_list
        self.theta_list = opt.theta_list

        self.size = len(self.theta_list) * len(self.phi_list)

        self.cx = self.H / 2
        self.cy = self.W / 2

        self.R_path = R_path

    def collate(self, index):
        phi = self.phi_list[index[0] % len(self.phi_list)]
        theta = self.theta_list[int(index[0] / len(self.phi_list))]

        poses, dirs = circle_poses(self.device, radius=self.radius_range[0], theta=theta, phi=phi,
                                   angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front)

        # fixed focal
        fov = (self.fovy_range[1] + self.fovy_range[0]) / 2
        focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
        intrinsics = np.array([focal, focal, self.cy, self.cx])

        if self.R_path:
            R = np.load(self.R_path)
            R = torch.tensor(R).to(self.device).float()
            R = torch.inverse(R)
            poses = R @ poses

        rays = get_rays(poses, intrinsics, self.H, self.W, -1)

        data = {
            'H': self.H,
            'W': self.W,
            'phi': phi,
            'theta': theta,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'dir': dirs,
        }

        return data

    def dataloader(self):
        loader = DataLoader(list(range(self.size)), batch_size=1, collate_fn=self.collate, shuffle=False,
                            num_workers=0)
        return loader
