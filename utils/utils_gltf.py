import os
import numpy as np
import torch
import pygltflib

class GLTFHelper:
    def __init__(self, faces, export_immediately = True, 
                 save_dir = None, fps = 30, frame_strip = 1,
                 prefix = ""):
        '''
        如果export_immediately为True, 则直接输出导入的batch;  
        否则, 调用export函数的时候需要提供target_frame_id, 当提供的frame_id到达它的时候才会一起导出.
        '''
        self.faces = faces.detach().cpu() if isinstance(faces, torch.Tensor) else faces
        self.immediate = export_immediately
        self.vbuf = torch.empty(0)
        self.save_dir, self.prefix = save_dir, prefix
        self.fps, self.strip = fps, frame_strip
        self._cnt = 0
        self._newest_frame_id = -1
        if self.save_dir is not None and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def export_ply_per_frame(self):
        import trimesh
        plydir = os.path.join(self.save_dir, f"ply_{self._cnt:06d}")
        if not os.path.exists(plydir):
            os.makedirs(plydir)
        for i in range(self.vbuf.shape[0]):
            mesh = trimesh.Trimesh(vertices=self.vbuf[i], faces=self.faces)
            mesh.export(os.path.join(plydir, f"{i:06d}.ply"))

    def set_cnt(self, cnt):
        self._cnt = cnt

    def pop_all(self):
        self._pop_vbuf()

    def _pop_vbuf(self):
        assert self.vbuf.shape[0] == self._newest_frame_id + 1
        if self.vbuf.shape[0] <= 0:
            return
        frames_to_gltf_morph(self.vbuf, self.faces, None, self.fps, self.strip,
                       os.path.join(self.save_dir, f"{self.prefix}_{self._cnt:06d}.glb"))
        self._cnt += 1
        self.vbuf = torch.empty(0)
        self._newest_frame_id = -1
    
    def export(self, 
               vertices_frames:torch.Tensor, 
               frame_id:torch.Tensor = -1):
        '''
        当not export_immediately的时候，frame_id, target_frame_id才是必须的.  
        vertices_frames: (n_frames, n_vertices, 3); faces: (n_faces, 3), frame_id:(n_frames, )   
        如果vertices有bs, 即(bs, n_seq, n_vertices, 3)，只取每个bs的第一帧，也就是 [:,0]  
        '''
        if self.immediate:  # 不管那么多，直接取出每个batch的第一帧导出...
            vertices_frames = vertices_frames[:,0].detach().cpu() if len(vertices_frames.shape) == 4 else vertices_frames
            frames_to_gltf_morph(vertices_frames, faces=self.faces, v_template=None, fps=self.fps, frame_strip=self.strip, 
                            save_path=os.path.join(self.save_dir, f"{self.prefix}_{self._cnt:06d}.glb"))
            self._cnt += 1
            return

        if len(vertices_frames.shape) == 4:  # 有batch, 一个seq只有一个end...
            # 不断地从seq中取出第一帧...
            vertices_frames = vertices_frames[:,0].detach().cpu()
            frame_id = frame_id[:,0].detach().cpu()
            
        assert len(vertices_frames.shape) == 3, "vertices_frames should be (n_frames, n_vertices, 3)"
        nseq = vertices_frames.shape[0]
        # print(vertices_frames.shape, frame_id)
        for i in range(nseq):
            this_frame = vertices_frames[i:i+1].detach().cpu()
            fid = frame_id[i].item()
            if fid > self._newest_frame_id:
                self.vbuf = torch.concat([self.vbuf, this_frame], dim=0)
                self._newest_frame_id = fid
            else:
                self._pop_vbuf()
                self.vbuf = torch.concat([self.vbuf, this_frame], dim=0)
                self._newest_frame_id = fid


def frames_to_gltf_morph(
        vertices_frames:torch.Tensor | np.ndarray,
        faces: torch.Tensor | np.ndarray,
        v_template: torch.Tensor | np.ndarray,
        fps: int = 30,
        frame_strip: int = 1,
        save_path : str = None
    ):
    '''
    vertices_frames: (n_frames, n_vertices, 3),  
    faces: (n_faces, 3),  
    v_template: (n_vertices, 3), can be None
    '''
    n_frames, n_vertices, _ = vertices_frames.shape
    n_faces, _ = faces.shape
    if v_template is None:
        v_template = vertices_frames[0]
    if isinstance(v_template, torch.Tensor):
        v_template = v_template.detach().cpu().numpy().astype(np.float32)
    if isinstance(vertices_frames, torch.Tensor):
        vertices_frames = vertices_frames.detach().cpu().numpy().astype(np.float32)
    if isinstance(faces,torch.Tensor):
        faces = faces.detach().cpu().numpy()
    vertices_frames = vertices_frames.astype(np.float32)
    faces = faces.astype(np.uint32)
    v_template = v_template.astype(np.float32)
    vertices_frames = vertices_frames - v_template   # morph targets是displacement!
    
    # create key times array
    times = np.arange(0, n_frames, dtype=np.float32) * frame_strip  # Keyframes int.
    times = times / fps  # Convert frame indices to seconds

    # create weights array
    weights = np.eye(n_frames, dtype=np.float32).flatten()

    # convert to bytes
    vertices_frames_blob = [vertices_frames[i].flatten().tobytes() for i in range(n_frames)]
    faces_blob = faces.flatten().tobytes()
    v_template_blob = v_template.flatten().tobytes()
    times_blob = times.flatten().tobytes()
    weights_blob = weights.flatten().tobytes()

    lb_faces = len(faces_blob)
    lb_total_v = sum(len(vblob) for vblob in vertices_frames_blob)
    lb_single_v = len(vertices_frames_blob[0])

    # Create GLTF2 object
    gltf = pygltflib.GLTF2()

    # Buffers
    gltf.buffers.append(
        pygltflib.Buffer(
            byteLength=lb_faces + lb_single_v + lb_total_v + len(times_blob) + len(weights_blob)  # faces + vtemp + vframes + times + weights
        )
    )

    # BufferViews and accessors of vertices and faces
    ## bufferview of faces indices
    gltf.bufferViews.append(
        pygltflib.BufferView(
            buffer=0,
            byteLength=lb_faces,
            target=pygltflib.ELEMENT_ARRAY_BUFFER,
        )
    )
    ## bufferview of vertices template
    gltf.bufferViews.append(
        pygltflib.BufferView(
            buffer=0,
            byteOffset=lb_faces,  # 跳过faces.
            byteLength=lb_single_v,
            target=pygltflib.ARRAY_BUFFER,
        )
    )
    ## bufferview of vertices
    for i, frame_blob in enumerate(vertices_frames_blob):
        gltf.bufferViews.append(
            pygltflib.BufferView(
                buffer=0,
                byteOffset=lb_faces + lb_single_v + i * lb_single_v,  # 跳过 faces, v_template, 前i个vframe
                byteLength=lb_single_v,
                target=pygltflib.ARRAY_BUFFER,
            )
        )

    ## accessor of faces indices
    gltf.accessors.append(
        pygltflib.Accessor(
            bufferView=0,
            componentType=pygltflib.UNSIGNED_INT,
            count=len(faces.flatten()),
            type=pygltflib.SCALAR,
            max=[int(faces.max())],
            min=[int(faces.min())],
        )
    )

    ## accessor of vtmp
    gltf.accessors.append(
        pygltflib.Accessor(
            bufferView=1,
            componentType=pygltflib.FLOAT,
            count = n_vertices,
            type=pygltflib.VEC3,
            max=v_template.max(axis=0).tolist(),
            min=v_template.min(axis=0).tolist(),
        )
    )

    ## accessor of morph targets
    for i, frame in enumerate(vertices_frames):
        gltf.accessors.append(
            pygltflib.Accessor(
                bufferView = 2 + i,
                componentType=pygltflib.FLOAT,
                count = n_vertices,
                type=pygltflib.VEC3,
                max=frame.max(axis=0).tolist(),
                min=frame.min(axis=0).tolist(),
            )
        )

    # Meshes
    gltf.meshes.append(
        pygltflib.Mesh(
            primitives=[
                pygltflib.Primitive(
                    attributes={'POSITION': 1},   # base template
                    indices = 0,
                    targets=[{'POSITION': 2 + i} for i in range(n_frames)]  # morph targets.
                )
            ]
        )
    )

    # Nodes
    gltf.nodes.append(pygltflib.Node(mesh=0))

    # Scenes
    gltf.scenes.append(pygltflib.Scene(nodes=[0]))

    # Update buffer views and accessors for time and weights
    gltf.bufferViews.append(
        pygltflib.BufferView(
            buffer=0,
            byteOffset=lb_faces + lb_single_v + lb_total_v,
            byteLength=len(times_blob),
        )
    )

    gltf.bufferViews.append(
        pygltflib.BufferView(
            buffer=0,
            byteOffset = lb_faces + lb_single_v + lb_total_v + len(times_blob),
            byteLength = len(weights_blob),
        )
    )

    gltf.accessors.append(
        pygltflib.Accessor(
            bufferView=len(gltf.bufferViews) - 2,
            componentType=pygltflib.FLOAT,
            count=len(times),
            type=pygltflib.SCALAR,
            max=[float(times.max())],
            min=[float(times.min())],
        )
    )

    gltf.accessors.append(
        pygltflib.Accessor(
            bufferView=len(gltf.bufferViews) - 1,
            componentType=pygltflib.FLOAT,
            count=weights.shape[0],
            type=pygltflib.SCALAR,
            max=[1.0],
            min=[0.0],
        )
    )

    # Animations
    gltf.animations.append(
        pygltflib.Animation(
            channels=[
                pygltflib.AnimationChannel(
                    sampler=0,
                    target=pygltflib.AnimationChannelTarget(
                        node=0,
                        path=pygltflib.WEIGHTS
                    )
                )
            ],
            samplers=[
                pygltflib.AnimationSampler(
                    input=len(gltf.accessors) - 2,  # Time accessor
                    output=len(gltf.accessors) - 1,  # Weights accessor
                    interpolation=pygltflib.ANIM_LINEAR
                )
            ]
        )
    )

    # Combine all binary data
    gltf.set_binary_blob(faces_blob + v_template_blob + b''.join(vertices_frames_blob) + times_blob + weights_blob)

    if save_path is not None:
        if not save_path.endswith('.glb'):
            save_path += ".glb"
        gltf.save_binary(save_path)
    
    return gltf


def frames_to_gltf_trans_rot(
        vertices:torch.Tensor | np.ndarray,
        faces: torch.Tensor | np.ndarray,
        transl: torch.Tensor | np.ndarray,
        rot: torch.Tensor | np.ndarray,
        fps: int = 30,
        frame_strip: int = 1,
        save_path : str = None
    ):
    '''
    vertices: (n_vertices, 3),  
    faces: (n_faces, 3),  
    transl: (nframes, 3), canbe none  
    rot: (nframes, 3, 3)  or (nframes, 4), (rotmat or quaternions), canbe none  
    注意旋转矩阵需要是左乘的旋转矩阵，即向量在矩阵右边.  
    '''
    n_vertices, _ = vertices.shape
    n_faces, _ = faces.shape
    if transl is not None:
        transl_anim = True
        n_frames = transl.shape[0]
    if rot is not None:
        rot_anim = True
        n_frames = rot.shape[0]
    if transl_anim and rot_anim:
        assert n_frames == transl.shape[0] == rot.shape[0]

    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy().astype(np.float32)
    if isinstance(faces,torch.Tensor):
        faces = faces.detach().cpu().numpy()
    if isinstance(transl, torch.Tensor):
        transl = transl.detach().cpu().numpy().astype(np.float32)
    if isinstance(rot, torch.Tensor):
        rot = rot.detach().cpu().numpy().astype(np.float32)
    vertices = vertices.astype(np.float32)
    faces = faces.astype(np.uint32)
    if rot.shape[1:] == (3, 3):  # rotmat
        from scipy.spatial.transform import Rotation as R
        r = R.from_matrix(rot)
        rot = r.as_quat()
    rot = rot.astype(np.float32)
    transl = transl.astype(np.float32)
    
    # create key times array
    if transl_anim or rot_anim:
        times = np.arange(0, n_frames, dtype=np.float32) * frame_strip  # Keyframes int.
        times = times / fps  # Convert frame indices to seconds

    # convert to bytes
    vertices_blob = vertices.flatten().tobytes()
    faces_blob = faces.flatten().tobytes()
    times_blob = times.flatten().tobytes() if transl_anim or rot_anim else b''
    transl_blob = transl.flatten().tobytes() if transl_anim else b''
    rot_blob = rot.flatten().tobytes() if rot_anim else b''

    lb_faces = len(faces_blob)
    lb_verts = len(vertices_blob)
    lb_times = len(times_blob)
    lb_transl = len(transl_blob)
    lb_rot = len(rot_blob)

    # Create GLTF2 object
    gltf = pygltflib.GLTF2()

    # Buffers
    gltf.buffers.append(
        pygltflib.Buffer(
            byteLength=lb_faces + lb_verts + lb_times + lb_transl + lb_rot  # faces + vtemp + vframes + times + weights
        )
    )

    # BufferViews and accessors of vertices and faces
    ## bufferview of faces indices
    gltf.bufferViews.append(
        pygltflib.BufferView(
            buffer=0,
            byteLength=lb_faces,
            target=pygltflib.ELEMENT_ARRAY_BUFFER,
        )
    )
    ## bufferview of vertices template
    gltf.bufferViews.append(
        pygltflib.BufferView(
            buffer=0,
            byteOffset=lb_faces,  # 跳过faces.
            byteLength=lb_verts,
            target=pygltflib.ARRAY_BUFFER,
        )
    )

    ## accessor of faces indices
    gltf.accessors.append(
        pygltflib.Accessor(
            bufferView=0,
            componentType=pygltflib.UNSIGNED_INT,
            count=len(faces.flatten()),
            type=pygltflib.SCALAR,
            max=[int(faces.max())],
            min=[int(faces.min())],
        )
    )

    ## accessor of vertices
    gltf.accessors.append(
        pygltflib.Accessor(
            bufferView=1,
            componentType=pygltflib.FLOAT,
            count = n_vertices,
            type=pygltflib.VEC3,
            max=vertices.max(axis=0).tolist(),
            min=vertices.min(axis=0).tolist(),
        )
    )

    # Meshes
    gltf.meshes.append(
        pygltflib.Mesh(
            primitives=[
                pygltflib.Primitive(
                    attributes={'POSITION': 1},   # base template
                    indices = 0,
                )
            ]
        )
    )

    # Nodes
    gltf.nodes.append(pygltflib.Node(mesh=0))

    # Scenes
    gltf.scenes.append(pygltflib.Scene(nodes=[0]))

    # Update buffer views and accessors for time, transl and rot
    ## bufferview of times
    if transl_anim or rot_anim:
        gltf.bufferViews.append(
            pygltflib.BufferView(
                buffer=0,
                byteOffset=lb_faces + lb_verts,
                byteLength=lb_times,
            )
        )
        gltf.accessors.append(
            pygltflib.Accessor(
                bufferView=len(gltf.bufferViews) - 1,
                componentType=pygltflib.FLOAT,
                count=len(times),
                type=pygltflib.SCALAR,
                max=[float(times.max())],
                min=[float(times.min())],
            )
        )

    if transl_anim:
        gltf.bufferViews.append(
            pygltflib.BufferView(
                buffer=0,
                byteOffset = lb_faces + lb_verts + lb_times,
                byteLength = lb_transl,
            )
        )
        gltf.accessors.append(
            pygltflib.Accessor(
                bufferView=len(gltf.bufferViews) - 1,
                componentType=pygltflib.FLOAT,
                count=len(times),
                type=pygltflib.VEC3,
                max=transl.max(axis=0).tolist(),
                min=transl.min(axis=0).tolist(),
            )
        )

    if rot_anim:
        gltf.bufferViews.append(
            pygltflib.BufferView(
                buffer=0,
                byteOffset = lb_faces + lb_verts + lb_times + (lb_transl if transl_anim else 0),
                byteLength = lb_rot,
            )
        )
        gltf.accessors.append(
            pygltflib.Accessor(
                bufferView=len(gltf.bufferViews) - 1,
                componentType=pygltflib.FLOAT,
                count=len(times),
                type=pygltflib.VEC4,
                max=rot.max(axis=0).tolist(),
                min=rot.min(axis=0).tolist(),
            )
        )

    if transl_anim or rot_anim:
        num_anim_channels = int(transl_anim) + int(rot_anim)
        time_accessor_idx = len(gltf.accessors) - num_anim_channels - 1
        pos = [pygltflib.TRANSLATION, pygltflib.ROTATION]
        targetlist = [pos[idx] for idx, val in enumerate([transl_anim, rot_anim]) if val]
        anim_samplers = [
            pygltflib.AnimationSampler(
                input=time_accessor_idx,  # Time accessor
                output=len(gltf.accessors) - num_anim_channels + i,  # Weights accessor
                interpolation=pygltflib.ANIM_LINEAR
            ) for i in range(num_anim_channels)
        ]
        anim_channels = [
            pygltflib.AnimationChannel(
                sampler=i,
                target=pygltflib.AnimationChannelTarget(
                    node=0,
                    path=targetlist[i]
                )
            ) for i in range(num_anim_channels)
        ]

        # Animations
        gltf.animations.append(
            pygltflib.Animation(
                channels=anim_channels,
                samplers=anim_samplers,
            )
        )

    # Combine all binary data
    gltf.set_binary_blob(faces_blob + vertices_blob + times_blob + transl_blob + rot_blob)

    if save_path is not None:
        if not save_path.endswith('.glb'):
            save_path += ".glb"
        gltf.save_binary(save_path)
    
    return gltf