import argparse
import tqdm
import os
import sys
import torch
import numpy as np
import open3d as o3d
import itertools

import pymeshlab
from PIL import Image
import torch.nn.functional as F
from diffusers import StableDiffusionImg2ImgPipeline

from collections import defaultdict

from daam import trace

sys.path.append('..')
from nerf.provider import get_rays, circle_poses



def generate_att_map(opt):
    device = 'cuda'
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(opt.sd_path)
    pipe = pipe.to(device)

    att_map_dict = {}

    for radius, fovy, phi, theta in itertools.product(opt.radius_list, opt.fovy_list, opt.phi_list,
                                                      opt.theta_list):

        phi = int(phi)
        theta = int(theta)
        print("{}_{}_{}_{}.png".format(radius, fovy, phi, theta))

        img = Image.open(os.path.join(opt.output_images_path, "{}_{}.png".format(theta, phi)))
        ave_map = torch.zeros((opt.mask_size, opt.mask_size)).cuda()
        count = 10

        for time in range(count):
            with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
                with trace(pipe) as tc:
                    out = pipe(opt.prompt, img, strength=0.75, num_inference_steps=50, guidance_scale=7.5)
                    heat_map = tc.compute_global_heat_map()
                    heat_map = heat_map.compute_word_heat_map(opt.key_word)
                    im = heat_map.heatmap.unsqueeze(0).unsqueeze(0)
                    im = F.interpolate(im.float().detach(), size=(opt.mask_size, opt.mask_size), mode='bicubic')
                    im = (im - im.min()) / (im.max() - im.min() + 1e-8)
                    # im = (im > 0.4).float()
                    ave_map += im[0][0]

        ave_map = ave_map / count
        ave_map = ave_map / ave_map.max()
        im = ave_map.cpu().detach().squeeze()

        im_np = im.numpy()

        att_map_dict["{}_{}.png".format(theta, phi)] = im_np

    np.save(os.path.join(opt.workspace_path, 'att_map.npy'), att_map_dict)


def back_projection(opt, scene):
    hit_tri_id_Set = set()
    att_map_dict = np.load(os.path.join(opt.workspace_path, 'att_map.npy'), allow_pickle=True).item()

    for radius, fovy, phi, theta in itertools.product(opt.radius_list, opt.fovy_list, opt.phi_list,
                                                      opt.theta_list):
        phi = int(phi)
        theta = int(theta)
        mask = np.array(att_map_dict["{}_{}.png".format(theta, phi)])
        mask[mask > opt.threshold] = 255
        mask = mask.reshape(-1)
        mask_idx = np.where(mask == 255)[0]

        H = opt.mask_size
        W = opt.mask_size

        cx = H / 2
        cy = W / 2

        poses, dirs = circle_poses('cpu', radius=radius, theta=theta, phi=phi)

        if len(opt.orient_r_path) > 0:
            R = np.load(opt.orient_r_path)
            R = torch.tensor(R).float()
            R = torch.inverse(R)
            poses = R @ poses

        focal = H / (2 * np.tan(np.deg2rad(fovy) / 2))
        intrinsics = np.array([focal, focal, cx, cy])

        rays = get_rays(poses, intrinsics, H, W, -1)

        origins = rays['rays_o'][0].numpy().reshape(-1, 3)
        directions = rays['rays_d'][0].numpy().reshape(-1, 3)

        for pixel_id in mask_idx:
            pixel_ori = origins[pixel_id].reshape(-1, 3)
            pixel_dir = directions[pixel_id].reshape(-1, 3)

            rays_core = np.concatenate([pixel_ori, pixel_dir], axis=1)
            rays = o3d.core.Tensor([rays_core], dtype=o3d.core.Dtype.Float32)
            ans = scene.cast_rays(rays)
            if ans['primitive_ids'].numpy()[0, 0] < 4294967295:
                hit_tri_id_Set.add(ans['primitive_ids'].numpy()[0, 0])
    return hit_tri_id_Set

def refine_region(opt, mesh, hit_tri_id_all_set):
    chosen_vec_set = set()
    mesh_tri_np = np.asarray(mesh.triangles)
    np_vec = np.asarray(mesh.vertices)

    for hit_tri_id in list(hit_tri_id_all_set):
        try:
            for vec_id in mesh_tri_np[hit_tri_id]:
                chosen_vec_set.add(vec_id)
        except BaseException:
            print('error', hit_tri_id)

    chosen_vec_list = list(chosen_vec_set)
    chosen_vec_list = sorted(chosen_vec_list)

    new_vec_np = np.zeros((len(chosen_vec_set), 3))
    for i, chosen_vec_id in enumerate(chosen_vec_list):
        new_vec_np[i] = np_vec[chosen_vec_id]

    new_tri = np.zeros((len(hit_tri_id_all_set), 3))
    for i, hit_tri_id in enumerate(list(hit_tri_id_all_set)):
        for j, vec_id in enumerate(mesh_tri_np[hit_tri_id]):
            new_tri[i][j] = chosen_vec_list.index(vec_id)

    new_vec_np = new_vec_np.astype(np.float32)
    mesh.vertices = o3d.utility.Vector3dVector(new_vec_np)

    new_tri = new_tri.astype(np.int32)
    mesh.triangles = o3d.utility.Vector3iVector(new_tri)

    o3d.io.write_triangle_mesh(os.path.join(opt.workspace_path, 'editing_region_mask.ply'), mesh)

    # refine step1
    face_nums = new_tri.shape[0]
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(os.path.join(opt.workspace_path, 'editing_region_mask.ply'))
    ms.meshing_remove_connected_component_by_face_number(mincomponentsize=int(face_nums * 0.25))
    pymesh = ms.current_mesh()
    new_mask_mesh = o3d.geometry.TriangleMesh()
    new_mask_mesh.vertices = o3d.utility.Vector3dVector(pymesh.vertex_matrix())
    new_mask_mesh.triangles = o3d.utility.Vector3iVector(pymesh.face_matrix())
    o3d.io.write_triangle_mesh(os.path.join(opt.workspace_path, 'editing_region_mask_refine1.ply'), new_mask_mesh)

    # refine step2
    scence_mesh = o3d.io.read_triangle_mesh(opt.mesh_path)
    np_tri = np.asarray(scence_mesh.triangles)
    np_vec = np.asarray(scence_mesh.vertices)

    mask_mesh = o3d.io.read_triangle_mesh(os.path.join(opt.workspace_path, 'editing_region_mask_refine1.ply'))
    alL_vertices = np.asarray(scence_mesh.vertices)
    edit_vertices = np.asarray(mask_mesh.vertices)
    edit_triangles = np.asarray(mask_mesh.triangles)

    edit_vertices_list = []

    for v in edit_vertices:
        edit_vertices_list.append(list(v))

    mask = [0] * int(alL_vertices.shape[0])
    mask_np = np.array(mask)
    for i, v in tqdm.tqdm(enumerate(alL_vertices)):
        if list(v) in edit_vertices_list:
            mask_np[i] = 1
        else:
            mask_np[i] = 0
    # register adjacent node
    vec_adjacent = defaultdict(set)
    for tri in np_tri:
        vec_adjacent[tri[0]].add(tri[1])
        vec_adjacent[tri[0]].add(tri[2])
        vec_adjacent[tri[1]].add(tri[0])
        vec_adjacent[tri[1]].add(tri[2])
        vec_adjacent[tri[2]].add(tri[0])
        vec_adjacent[tri[2]].add(tri[1])

    vec_list = []
    for i in range(mask_np.shape[0]):
        if mask_np[i] == 1:
            vec_list += vec_adjacent[i]
    vec_list = set(vec_list)
    vec_list = list(vec_list)

    for vec_id in tqdm.tqdm(vec_list):
        if mask_np[vec_id] == 0:
            adjacent = vec_adjacent[vec_id]
            count = 0
            occ_list = []
            while count < opt.max_depth:
                new_list = []
                for tmp_id in adjacent:
                    occ_list.append(tmp_id)
                    if mask_np[tmp_id] == 0:
                        for j in vec_adjacent[tmp_id]:
                            if j not in occ_list:
                                new_list.append(j)
                new_list = set(new_list)
                new_list = list(new_list)
                count += 1
                adjacent = new_list
            in_count = 0
            for tmp_id in adjacent:
                if mask_np[tmp_id] == 0:
                    in_count += 1
            if in_count == 0:
                mask_np[vec_id] = 1
                for id in occ_list:
                    mask_np[occ_list] = 1
    chosen_vec_set = set()
    for i in range(mask_np.shape[0]):
        if mask_np[i] == 1:
            chosen_vec_set.add(i)

    hit_tri_id_all_set = set()

    for tri_id, tri in enumerate(np_tri):
        if mask_np[tri[0]] == 1 and mask_np[tri[1]] == 1 and mask_np[tri[2]] == 1:
            hit_tri_id_all_set.add(tri_id)

    chosen_vec_list = list(chosen_vec_set)
    chosen_vec_list = sorted(chosen_vec_list)

    chosen_vec_dict = {}

    for i, chosen_vec_id in enumerate(chosen_vec_list):
        chosen_vec_dict[chosen_vec_id] = i

    new_vec_np = np.zeros((len(chosen_vec_set), 3))
    for i, chosen_vec_id in enumerate(chosen_vec_list):
        new_vec_np[i] = np_vec[chosen_vec_id]

    new_tri = np.zeros((len(hit_tri_id_all_set), 3))
    for i, hit_tri_id in tqdm.tqdm(enumerate(list(hit_tri_id_all_set))):
        for j, vec_id in enumerate(np_tri[hit_tri_id]):
            new_tri[i][j] = chosen_vec_dict[vec_id]

    new_vec_np = new_vec_np.astype(np.float32)
    mesh.vertices = o3d.utility.Vector3dVector(new_vec_np)

    new_tri = new_tri.astype(np.int32)
    mesh.triangles = o3d.utility.Vector3iVector(new_tri)

    o3d.io.write_triangle_mesh(os.path.join(opt.workspace_path, 'editing_region_mask_refine2.ply'), mesh)

    return mask_np
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_images_path', default=None, help="text prompt")
    parser.add_argument('--workspace_path', default='', type=str, help="negative text prompt")
    parser.add_argument('--radius_list', type=float, nargs='*', default=[0.5])
    parser.add_argument('--fovy_list', type=float, nargs='*', default=[50])
    parser.add_argument('--phi_list', type=float, nargs='*', default=[0])
    parser.add_argument('--theta_list', type=float, nargs='*', default=[75])
    parser.add_argument('--mesh_path',  type=str, default=None)
    parser.add_argument('--orient_r_path', type=str, default=None)
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--key_word', type=str, default=None)
    parser.add_argument('--sd_path', type=str, default=None)
    parser.add_argument('--threshold', type=float, default=0.75)
    parser.add_argument('--mask_size', type=int, default=1024)
    parser.add_argument('--max_depth', type=int, default=10)
    opt = parser.parse_args()


    """Main function."""

    os.makedirs(opt.workspace_path, exist_ok=True)

    # generate att map

    generate_att_map(opt)

    # back projection to mesh
    print(f"back projection to mesh")
    mesh = o3d.io.read_triangle_mesh(opt.mesh_path)
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)

    hit_tri_id_Set = back_projection(opt, scene)

    # refine_region
    print(f"refine editing region")
    mask_np = refine_region(opt, mesh, hit_tri_id_Set)

    mesh = o3d.io.read_triangle_mesh(opt.mesh_path)
    mesh.paint_uniform_color([1, 1, 1])
    np_color = np.asarray(mesh.vertex_colors)
    np_color[mask_np == 1] = [1, 0, 0]
    o3d.io.write_triangle_mesh(os.path.join(opt.workspace_path, 'color_editing_region.ply'), mesh)

    os.remove(os.path.join(opt.workspace_path, 'editing_region_mask.ply'))
    os.remove(os.path.join(opt.workspace_path, 'editing_region_mask_refine1.ply'))
    os.rename(os.path.join(opt.workspace_path, 'editing_region_mask_refine2.ply'),
              os.path.join(opt.workspace_path, 'editing_region_mask.ply'))

    # convert mask.ply to numpy information
    mesh = o3d.io.read_triangle_mesh(opt.mesh_path)
    mask_mesh = o3d.io.read_triangle_mesh(os.path.join(opt.workspace_path, 'editing_region_mask.ply'))

    alL_vertices = np.asarray(mesh.vertices)
    edit_vertices = np.asarray(mask_mesh.vertices)
    edit_triangles = np.asarray(mask_mesh.triangles)
    pymesh = pymeshlab.Mesh(edit_vertices, edit_triangles)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymesh)
    ms.compute_selection_from_mesh_border()
    pymesh = ms.current_mesh()
    edge_mask = pymesh.vertex_selection_array() * 1

    edit_vertices_list = []
    for i, v in enumerate(edit_vertices):
        edit_vertices_list.append(list(v))
    edit_vertices_list_noedge = []
    for i, v in enumerate(edit_vertices):
        if edge_mask[i] == 0:
            edit_vertices_list_noedge.append(list(v))
    mesh_info = {}

    edited_flag_list = []
    for i, v in tqdm.tqdm(enumerate(alL_vertices)):
        if list(v) in edit_vertices_list:
            edited_flag_list.append(1)
        else:
            edited_flag_list.append(0)
    mesh_info['editing_mask'] = np.array(edited_flag_list)
    print(f"vertices in editing region sum : {sum(edited_flag_list)}")

    edited_flag_list = []
    for i, v in tqdm.tqdm(enumerate(alL_vertices)):
        if list(v) in edit_vertices_list_noedge:
            edited_flag_list.append(1)
        else:
            edited_flag_list.append(0)
    mesh_info['editing_mask_noedge'] = np.array(edited_flag_list)
    print(f"vertices in noedge editing region sum : {sum(edited_flag_list)}")

    print(f"Saved locating results to: {opt.workspace_path}")
    np.save(os.path.join(opt.workspace_path, 'editing_region_info.npy'), mesh_info)

