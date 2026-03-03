# lib/localization_engine.py
import numpy as np
import cv2
import time
import torch
import h5py
import pycolmap
import json
import threading  # [New] 用於執行緒同步
from pathlib import Path
from scipy.spatial.transform import Rotation
from unittest.mock import patch

# HLOC Imports
from hloc import extractors, matchers, extract_features, match_features
from hloc.utils.base_model import dynamic_load

# [Modified] 引入 colmap_to_scipy_quat 以便進行座標轉換
try:
    from .map_utils import compute_sim2_transform, colmap_to_scipy_quat
except ImportError:
    from map_utils import compute_sim2_transform, colmap_to_scipy_quat

class LocalizationEngine:
    def __init__(self, project_root: Path, config_path: Path, anchors_path: Path, outputs_dir: Path = None, device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # [New] 使用訊號量限制同時運算的數量為 1
        # 這能防止多個請求同時進入 GPU 導致顯存溢出 (OOM)，並解決 HDF5 執行緒安全問題
        self.semaphore = threading.BoundedSemaphore(1)
        
        print(f"[Init] LocalizationEngine using device: {self.device}")
        
        self.config = {}
        if config_path.exists():
            with open(config_path) as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        k, v = line.strip().split('=', 1)
                        self.config[k.strip()] = v.strip().strip('"').strip("'")
        
        self.global_conf_name = self.config.get("GLOBAL_CONF", "netvlad")
        self.default_fov = float(self.config.get("FOV_QUERY", self.config.get("FOV", 69.4)))
        self.default_top_k = int(self.config.get("TOP_K_DB", 10))

        self.project_root = project_root
        
        # [Offline Fix] 設定離線模型路徑
        CACHE_ROOT = Path("/root/.cache/torch/hub")
        SUPERPOINT_WEIGHTS = CACHE_ROOT / "checkpoints/superpoint_lightglue_v0-1_arxiv.pth"
        MEGALOC_REPO = CACHE_ROOT / "gmberton_MegaLoc_main" 
        DINOV2_REPO = CACHE_ROOT / "facebookresearch_dinov2_main" 

        print("[Init] Loading Neural Network Models (Offline Mode)...")

        _real_hub_load = torch.hub.load
        _real_load_url = torch.hub.load_state_dict_from_url
        
        def mock_hub_load(repo_or_dir, *args, **kwargs):
            if repo_or_dir == "gmberton/MegaLoc" and MEGALOC_REPO.exists():
                print(f"  [Offline] Redirecting MegaLoc to local repo: {MEGALOC_REPO}")
                kwargs['source'] = 'local'
                return _real_hub_load(str(MEGALOC_REPO), *args, **kwargs)
            elif repo_or_dir == "facebookresearch/dinov2":
                if DINOV2_REPO.exists():
                    print(f"  [Offline] Redirecting DINOv2 to local repo: {DINOV2_REPO}")
                    kwargs['source'] = 'local'
                    return _real_hub_load(str(DINOV2_REPO), *args, **kwargs)
            return _real_hub_load(repo_or_dir, *args, **kwargs)

        def mock_load_url(url, *args, **kwargs):
            url_str = str(url).lower()
            if "superpoint" in url_str:
                print(f"  [Offline] Intercepted SuperPoint weights URL, loading local: {SUPERPOINT_WEIGHTS}")
                if SUPERPOINT_WEIGHTS.exists():
                    return torch.load(SUPERPOINT_WEIGHTS, map_location=self.device)
            return _real_load_url(url, *args, **kwargs)

        with patch('torch.hub.load', side_effect=mock_hub_load), \
             patch('torch.hub.load_state_dict_from_url', side_effect=mock_load_url):
            
            local_conf = extract_features.confs['superpoint_aachen']
            ModelLocal = dynamic_load(extractors, local_conf['model']['name'])
            self.model_extract_local = ModelLocal(local_conf['model']).eval().to(self.device)
            
            global_conf = extract_features.confs[self.global_conf_name]
            ModelGlobal = dynamic_load(extractors, global_conf['model']['name'])
            self.model_extract_global = ModelGlobal(global_conf['model']).eval().to(self.device)
            
            matcher_conf = match_features.confs['superpoint+lightglue']
            ModelMatcher = dynamic_load(matchers, matcher_conf['model']['name'])
            self.model_matcher = ModelMatcher(matcher_conf['model']).eval().to(self.device)
        
        target_outputs = outputs_dir if outputs_dir else (project_root / "outputs-hloc")
        self.blocks = {}
        self._load_blocks(target_outputs, anchors_path)

    def _load_blocks(self, outputs_root, anchors_path):
        if not anchors_path.exists():
            raise FileNotFoundError(f"[Error] Anchors file missing at: {anchors_path}")

        with open(anchors_path, 'r') as f:
            anchors = json.load(f)
        
        anchors_keys = set(anchors.keys())
        print(f"[Init] Anchors defined for: {list(anchors_keys)}")

        physical_block_names = set()
        if outputs_root.exists():
            physical_block_names = {p.name for p in outputs_root.iterdir() if p.is_dir()}

        orphaned_blocks = physical_block_names - anchors_keys
        if orphaned_blocks:
            print("\n" + "?"*60)
            print(f"[WARNING] The following blocks exist on disk but are NOT in anchors.json:")
            for m in orphaned_blocks:
                print(f"  - {m} (Skipped loading)")
            print("?"*60 + "\n")

        failed_blocks = []

        for block_name in anchors_keys:
            block_dir = outputs_root / block_name
            if not block_dir.exists():
                failed_blocks.append(f"{block_name} (Directory not found)")
                continue

            sfm_dir = block_dir / "sfm_aligned"
            if not (sfm_dir / "images.bin").exists(): sfm_dir = block_dir / "sfm"
            
            global_h5 = block_dir / f"global-{self.global_conf_name}.h5"
            local_h5_path = block_dir / "local-superpoint_aachen.h5"
            
            if not (sfm_dir/"images.bin").exists() or not global_h5.exists() or not local_h5_path.exists():
                failed_blocks.append(f"{block_name} (Missing SFM or H5 files)")
                continue

            print(f"[Init] Loading Block: {block_name}")
            try:
                g_names = []
                g_vecs = []
                with h5py.File(global_h5, 'r') as f:
                    def visit(name, obj):
                        if isinstance(obj, h5py.Group) and 'global_descriptor' in obj:
                            g_names.append(name)
                            g_vecs.append(obj['global_descriptor'].__array__())
                    f.visititems(visit)
                
                if not g_vecs: 
                    failed_blocks.append(f"{block_name} (Global descriptors empty)")
                    continue

                # 全域特徵留在 CPU (不呼叫 .to(self.device))
                g_vecs = torch.from_numpy(np.array(g_vecs)).float().squeeze()
                if g_vecs.ndim == 1: g_vecs = g_vecs.unsqueeze(0)

                recon = pycolmap.Reconstruction(sfm_dir)
                name_to_id = {img.name: img_id for img_id, img in recon.images.items()}
                
                transform = None
                try:
                    transform = compute_sim2_transform(recon, anchors[block_name])
                    if transform:
                        s = transform['s']
                        theta_deg = np.degrees(transform['theta'])
                        print(f"    > Aligned: Scale={s:.4f}, Rot={theta_deg:.2f}°")
                    else:
                        print(f"    > [Warn] Failed to compute Sim2 transform for {block_name}")
                except Exception as e:
                    print(f"    > [Warn] Transform error: {e}")

                self.blocks[block_name] = {
                    'recon': recon,
                    'name_to_id': name_to_id,
                    'global_names': g_names,
                    'global_vecs': g_vecs,
                    'local_h5_path': local_h5_path,
                    'local_h5': h5py.File(local_h5_path, 'r'),
                    'transform': transform,
                    'block_root': block_dir
                }

            except Exception as e:
                failed_blocks.append(f"{block_name} (Load Exception: {e})")

        if failed_blocks:
            print("\n" + "!"*60)
            print(f"[WARNING] The following blocks are defined in Anchors but FAILED to load:")
            for msg in failed_blocks:
                print(f"  - {msg}")
            print("!"*60 + "\n")

    @torch.no_grad()
    def localize(self, 
                 image_arr: np.ndarray, 
                 fov_deg: float = None, 
                 return_details: bool = False, 
                 top_k_db: int = None,
                 verbose: bool = False,
                 block_filter: list = None):
        
        import time 
        
        def sync_time():
            if 'cuda' in str(self.device):
                torch.cuda.synchronize()
            return time.time()

        timings = {}
        t_start_total = sync_time()
        
        with self.semaphore:
            if fov_deg is None: fov_deg = self.default_fov
            if top_k_db is None: top_k_db = self.default_top_k
            
            # ==========================================
            # 1. 影像前處理 (Image Preprocessing)
            # ==========================================
            t0 = sync_time()
            h_orig, w_orig = image_arr.shape[:2]
            resize_max = 1024
            scale = 1.0
            new_w, new_h = w_orig, h_orig
            
            if max(h_orig, w_orig) >= resize_max:
                scale = resize_max / max(h_orig, w_orig)
                new_w, new_h = int(round(w_orig * scale)), int(round(h_orig * scale))
                image_tensor = cv2.resize(image_arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                image_tensor = image_arr

            scale_x = w_orig / new_w
            scale_y = h_orig / new_h
            
            img_t = torch.from_numpy(image_tensor.transpose(2, 0, 1)).float().div(255.).unsqueeze(0).to(self.device)
            img_g = torch.from_numpy(cv2.cvtColor(image_tensor, cv2.COLOR_RGB2GRAY)).float().div(255.).unsqueeze(0).unsqueeze(0).to(self.device)
            
            t1 = sync_time()
            timings['1_preprocessing'] = t1 - t0

            # ==========================================
            # 2. 特徵提取與相機初始化 (Feature Extraction)
            # ==========================================
            t2_0 = sync_time()
            q_global = self.model_extract_global({'image': img_t})['global_descriptor']
            t2_1 = sync_time()
            
            q_local = self.model_extract_local({'image': img_g})
            t2_2 = sync_time()
            
            q_global_cpu = q_global.cpu()   
            kpts = q_local['keypoints'][0]
            desc = q_local['descriptors'][0]
            
            if verbose:
                print(f"  [Log] Query Kpts: {len(kpts)} (Scale: {scale_x:.4f}, {scale_y:.4f})")
            
            diag = {
                'num_kpts': len(kpts),
                'pnp_top1_block': 'None', 'pnp_top1_inliers': 0, 
                'num_matches_2d': 0, 'num_matches_3d': 0, 
                'status': 'Fail_Unknown',
                'db_ranks': [],
                'pose_qw': "", 'pose_qx': "", 'pose_qy': "", 'pose_qz': "",
                'pose_tx': "", 'pose_ty': "", 'pose_tz': "",
                'map_x': "", 'map_y': "", 'map_yaw': ""
            }

            kpts[:, 0] = (kpts[:, 0] + 0.5) * scale_x
            kpts[:, 1] = (kpts[:, 1] + 0.5) * scale_y
            
            fov_rad = np.deg2rad(fov_deg)
            f = 0.5 * max(w_orig, h_orig) / np.tan(fov_rad / 2.0)
            camera = pycolmap.Camera(
                model='SIMPLE_PINHOLE', width=w_orig, height=h_orig, 
                params=np.array([f, w_orig/2.0, h_orig/2.0], dtype=np.float64)
            )
            
            t2_3 = sync_time()
            
            # 寫入細項時間
            timings['2a_feat_global_gpu'] = t2_1 - t2_0
            timings['2b_feat_local_gpu'] = t2_2 - t2_1
            timings['2c_feat_cpu_post'] = t2_3 - t2_2
            timings['2_feature_extraction'] = t2_3 - t2_0

            # ==========================================
            # 3. 全域檢索與區塊篩選 (Global Retrieval)
            # ==========================================
            candidate_blocks = []
            for name, block in self.blocks.items():
                if block_filter is not None and name not in block_filter:
                    continue
                sim = torch.matmul(q_global_cpu, block['global_vecs'].t())
                k_scoring = min(5, sim.shape[1])
                if k_scoring > 0:
                    topk_vals, _ = torch.topk(sim, k=k_scoring, dim=1)
                    score = torch.mean(topk_vals, dim=1)
                else:
                    score = torch.tensor([0.0], device=self.device)
                if score.item() > 0.01: candidate_blocks.append((score.item(), name, sim))
            
            candidate_blocks.sort(key=lambda x: x[0], reverse=True)
            candidate_blocks = candidate_blocks[:3]

            for i in range(3):
                rank = i + 1
                if i < len(candidate_blocks):
                    diag[f'retrieval_top{rank}'] = candidate_blocks[i][1]
                    diag[f'retrieval_score{rank}'] = candidate_blocks[i][0]
                else:
                    diag[f'retrieval_top{rank}'] = "None"
                    diag[f'retrieval_score{rank}'] = 0.0

            t3 = sync_time()
            timings['3_global_retrieval'] = t3 - t2_3

            if not candidate_blocks:
                diag['status'] = 'Fail_No_Retrieval'
                best_result = {'success': False, 'inliers': 0, 'diagnosis': diag}
                return self._finalize_result(best_result, timings, t_start_total, sync_time)

            valid_block_results = []
            best_fail_stats = diag.copy()

            # 初始化 Matching 細項時間累積
            time_match_h5_io = 0.0
            time_match_transfer_gpu = 0.0
            time_match_lightglue_gpu = 0.0
            time_match_cpu_post = 0.0
            time_local_matching_total = 0.0
            time_pnp_transform = 0.0

            for _, block_name, sim_matrix in candidate_blocks:
                block = self.blocks[block_name]
                k_val = min(top_k_db, sim_matrix.shape[1])
                scores, indices = torch.topk(sim_matrix, k=k_val, dim=1)
                indices = indices[0].cpu().numpy()
                
                # ==========================================
                # 4. 局部特徵匹配 (Local Feature Matching)
                # ==========================================
                t_match_start = sync_time()
                
                # (4a) H5 I/O 與 CPU 預載
                t_h5_start = sync_time()
                db_features_preloaded = []
                for rank, db_idx in enumerate(indices):
                    db_name = block['global_names'][db_idx]
                    if db_name not in block['local_h5'] or db_name not in block['name_to_id']:
                        continue
                    grp = block['local_h5'][db_name]
                    kpts_db = torch.from_numpy(grp['keypoints'].__array__()).float()
                    desc_db = torch.from_numpy(grp['descriptors'].__array__()).float()
                    if desc_db.shape[0] != 256 and desc_db.shape[1] == 256:
                        desc_db = desc_db.T
                    db_features_preloaded.append({
                        'rank': rank, 'db_name': db_name, 'kpts': kpts_db, 'desc': desc_db
                    })
                t_h5_end = sync_time()
                time_match_h5_io += (t_h5_end - t_h5_start)

                p2d_list, p3d_list = [], []
                viz_details = {}
                current_block_stats = {'matches_2d_sum': 0, 'matches_3d_sum': 0}
                current_db_ranks = [] 
                unique_matches = set()

                for feat in db_features_preloaded:
                    db_name = feat['db_name']
                    rank = feat['rank']
                    
                    # (4b) 移至 GPU 準備運算
                    t_transfer_start = sync_time()
                    kpts_db = feat['kpts'].to(self.device)
                    desc_db = feat['desc'].to(self.device)
                    img_obj = block['recon'].images[block['name_to_id'][db_name]]
                    cam_db = block['recon'].cameras[img_obj.camera_id]
                    data = {
                        'image0': torch.empty((1,1,h_orig,w_orig), device=self.device),
                        'keypoints0': kpts.unsqueeze(0), 'descriptors0': desc.unsqueeze(0),
                        'image1': torch.empty((1,1,cam_db.height,cam_db.width), device=self.device),
                        'keypoints1': kpts_db.unsqueeze(0), 'descriptors1': desc_db.unsqueeze(0)
                    }
                    t_transfer_end = sync_time()
                    time_match_transfer_gpu += (t_transfer_end - t_transfer_start)
                    
                    # (4c) LightGlue GPU 運算
                    t_lg_start = sync_time()
                    matches = self.model_matcher(data)['matches0'][0]
                    t_lg_end = sync_time()
                    time_match_lightglue_gpu += (t_lg_end - t_lg_start)

                    # (4d) GPU移至CPU、轉換、過濾重複點
                    t_cpu_start = sync_time()
                    valid = matches > -1
                    n_2d = valid.sum().item()
                    current_block_stats['matches_2d_sum'] += n_2d
                    
                    if rank < 3: current_db_ranks.append({'name': db_name, 'matches_2d': n_2d})
                    if n_2d < 4: 
                        t_cpu_end = sync_time()
                        time_match_cpu_post += (t_cpu_end - t_cpu_start)
                        continue

                    p3d_ids = np.array([p.point3D_id if p.has_point3D() else -1 for p in img_obj.points2D])
                    m_q = torch.where(valid)[0].cpu().numpy()
                    m_db = matches[valid].cpu().numpy()
                    valid_3d = m_db < len(p3d_ids)
                    m_q, m_db = m_q[valid_3d], m_db[valid_3d]
                    target_ids = p3d_ids[m_db]
                    has_3d = target_ids != -1
                    n_3d = int(has_3d.sum())
                    current_block_stats['matches_3d_sum'] += n_3d
                    
                    if n_3d < 4:
                        t_cpu_end = sync_time()
                        time_match_cpu_post += (t_cpu_end - t_cpu_start)
                        continue
                    
                    m_q_valid = m_q[has_3d]
                    target_ids_valid = target_ids[has_3d]
                    new_p2d, new_p3d = [], []
                    kpts_np = kpts.cpu().numpy()
                    points3D_map = block['recon'].points3D
                    
                    for q_idx, tid in zip(m_q_valid, target_ids_valid):
                        q_idx = int(q_idx)
                        tid = int(tid)
                        if (q_idx, tid) not in unique_matches:
                            unique_matches.add((q_idx, tid))
                            new_p2d.append(kpts_np[q_idx])
                            new_p3d.append(points3D_map[tid].xyz)
                    
                    if new_p2d:
                        p2d_list.append(np.array(new_p2d, dtype=np.float64))
                        p3d_list.append(np.array(new_p3d, dtype=np.float64))
                    
                    if rank == 0: 
                        viz_details['matched_db_name'] = db_name
                        if return_details:
                            viz_details.update({
                                'db_image_path': str(block['block_root'] / "images" / db_name),
                                'kpts_query': kpts_np,
                                'kpts_db': kpts_db.cpu().numpy(),
                                'matches': matches.cpu().numpy()
                            })
                    t_cpu_end = sync_time()
                    time_match_cpu_post += (t_cpu_end - t_cpu_start)

                t_match_end = sync_time()
                time_local_matching_total += (t_match_end - t_match_start)

                if block_name == diag.get('retrieval_top1'):
                     best_fail_stats['num_matches_2d'] = current_block_stats['matches_2d_sum']
                     best_fail_stats['num_matches_3d'] = current_block_stats['matches_3d_sum']
                     best_fail_stats['db_ranks'] = current_db_ranks
                     if current_block_stats['matches_3d_sum'] == 0: best_fail_stats['status'] = 'Fail_No_3D_Match'
                     else: best_fail_stats['status'] = 'Fail_PnP_Error'

                if not p2d_list: continue
                p2d_concat = np.concatenate(p2d_list, axis=0)
                p3d_concat = np.concatenate(p3d_list, axis=0)
                
                # ==========================================
                # 5. PnP 姿態估計與優化 (PnP Pose & Transform)
                # ==========================================
                t_pnp_start = sync_time()
                try:
                    refine_opts = pycolmap.AbsolutePoseRefinementOptions()
                    refine_opts.refine_focal_length = True
                    refine_opts.refine_extra_params = False
                    ret = pycolmap.estimate_and_refine_absolute_pose(
                        p2d_concat, p3d_concat, camera, 
                        estimation_options={'ransac': {'max_error': 12.0}},
                        refinement_options=refine_opts
                    )
                    success, qvec, tvec, num_inliers = False, None, None, 0
                    if ret:
                        if isinstance(ret, dict):
                            success = ret.get('success', False)
                            num_inliers = ret.get('num_inliers', 0)
                        else:
                            success = ret.success
                            num_inliers = ret.num_inliers

                        if not success:
                             ret_ransac = pycolmap.estimate_absolute_pose(
                                 p2d_concat, p3d_concat, camera, estimation_options={'ransac': {'max_error': 12.0}}
                             )
                             if ret_ransac:
                                if isinstance(ret_ransac, dict):
                                    if ret_ransac.get('num_inliers', 0) > 0: success = True
                                elif ret_ransac.num_inliers > 0: success = True
                                if success: ret = ret_ransac 

                        if success and num_inliers < 15: success = False

                        if success:
                            if isinstance(ret, dict):
                                if 'qvec' in ret: q_raw, tvec = ret['qvec'], ret['tvec']
                                elif 'cam_from_world' in ret:
                                    q_raw = ret['cam_from_world'].rotation.quat
                                    tvec = ret['cam_from_world'].translation
                            else:
                                if ret.cam_from_world:
                                    q_raw = ret.cam_from_world.rotation.quat
                                    tvec = ret.cam_from_world.translation
                            if q_raw is not None:
                                qvec = np.array([q_raw[3], q_raw[0], q_raw[1], q_raw[2]])

                    if success and qvec is not None:
                        q_scipy = colmap_to_scipy_quat(qvec)
                        R_w2c = Rotation.from_quat(q_scipy).as_matrix()
                        R_c2w = R_w2c.T
                        t_vec = np.array(tvec)
                        cam_center_sfm = -R_c2w @ t_vec
                        view_dir = R_c2w[:, 2]
                        sfm_yaw = np.degrees(np.arctan2(view_dir[1], view_dir[0]))
                        
                        trans = block.get('transform')
                        if trans:
                            s = trans['s']
                            theta = trans['theta']
                            t_map = trans['t']
                            R_map = trans['R']
                            p_sfm_2d = cam_center_sfm[:2]
                            p_map = s * (R_map @ p_sfm_2d) + t_map
                            map_x, map_y = float(p_map[0]), float(p_map[1])
                            m_yaw = sfm_yaw + np.degrees(theta)
                            map_yaw = float((m_yaw + 180) % 360 - 180)
                        else:
                            map_x, map_y, map_yaw = None, None, None

                        res_diag = diag.copy()
                        res_diag.update({
                            'pnp_top1_block': block_name, 'pnp_top1_inliers': num_inliers,
                            'num_matches_2d': current_block_stats['matches_2d_sum'], 'num_matches_3d': len(p2d_concat),
                            'status': 'Success', 'db_ranks': current_db_ranks,
                            'pose_qw': qvec[0], 'pose_qx': qvec[1], 'pose_qy': qvec[2], 'pose_qz': qvec[3],
                            'pose_tx': tvec[0], 'pose_ty': tvec[1], 'pose_tz': tvec[2],
                            'map_x': map_x, 'map_y': map_y, 'map_yaw': map_yaw
                        })
                        
                        res = {
                            'success': True, 'block': block_name, 'pose': {'qvec': qvec, 'tvec': tvec}, 
                            'transform': block['transform'], 'inliers': num_inliers,
                            'matched_db_name': viz_details.get('matched_db_name', 'unknown'),
                            'diagnosis': res_diag
                        }
                        if return_details: res.update(viz_details)
                        valid_block_results.append(res)
                except Exception as e:
                    print(f"[Error] PnP failed for {block_name}: {e}")
                
                t_pnp_end = sync_time()
                time_pnp_transform += (t_pnp_end - t_pnp_start)
            
            timings['4a_match_h5_io'] = time_match_h5_io
            timings['4b_match_transfer_gpu'] = time_match_transfer_gpu
            timings['4c_match_lightglue_gpu'] = time_match_lightglue_gpu
            timings['4d_match_cpu_post'] = time_match_cpu_post
            timings['4_local_matching'] = time_local_matching_total
            timings['5_pnp_and_transform'] = time_pnp_transform

            if valid_block_results:
                valid_block_results.sort(key=lambda x: x['inliers'], reverse=True)
                best_result = valid_block_results[0]
                
                if len(valid_block_results) > 1:
                    second = valid_block_results[1]
                    best_result['diagnosis']['pnp_top2_inliers'] = second['inliers']
                    best_result['diagnosis']['pnp_top2_block'] = second['block']
                else:
                    best_result['diagnosis']['pnp_top2_inliers'] = 0
                    best_result['diagnosis']['pnp_top2_block'] = "None"
                
                if len(valid_block_results) > 2:
                    third = valid_block_results[2]
                    best_result['diagnosis']['pnp_top3_inliers'] = third['inliers']
                    best_result['diagnosis']['pnp_top3_block'] = third['block']
                else:
                    best_result['diagnosis']['pnp_top3_inliers'] = 0
                    best_result['diagnosis']['pnp_top3_block'] = "None"
            else:
                best_result = {'success': False, 'inliers': 0, 'diagnosis': best_fail_stats}
                for r in ['1', '2', '3']: 
                    best_result['diagnosis'][f'pnp_top{r}_inliers'] = 0
                    best_result['diagnosis'][f'pnp_top{r}_block'] = "None"
            
            return self._finalize_result(best_result, timings, t_start_total, sync_time)

    # 封裝一個小函式處理結果與時間的綁定，保持主程式整潔
    def _finalize_result(self, best_result, timings, t_start_total, sync_time):
        timings['total_time'] = sync_time() - t_start_total
        diag = best_result.get('diagnosis', {})
        diag['time_preprocessing'] = timings.get('1_preprocessing', 0.0)
        
        # 特徵提取細項
        diag['time_feat_ext'] = timings.get('2_feature_extraction', 0.0)
        diag['time_feat_global_gpu'] = timings.get('2a_feat_global_gpu', 0.0)
        diag['time_feat_local_gpu'] = timings.get('2b_feat_local_gpu', 0.0)
        diag['time_feat_cpu'] = timings.get('2c_feat_cpu_post', 0.0)
        
        diag['time_global_retrieval'] = timings.get('3_global_retrieval', 0.0)
        
        # 匹配細項
        diag['time_local_matching'] = timings.get('4_local_matching', 0.0)
        diag['time_match_h5_io'] = timings.get('4a_match_h5_io', 0.0)
        diag['time_match_transfer_gpu'] = timings.get('4b_match_transfer_gpu', 0.0)
        diag['time_match_lg_gpu'] = timings.get('4c_match_lightglue_gpu', 0.0)
        diag['time_match_cpu'] = timings.get('4d_match_cpu_post', 0.0)
        
        diag['time_pnp_transform'] = timings.get('5_pnp_and_transform', 0.0)
        diag['time_total'] = timings.get('total_time', 0.0)
        
        best_result['diagnosis'] = diag
        best_result['timings'] = timings
        return best_result


    def format_diagnosis(self, diag_raw: dict = None) -> dict:
        if diag_raw is None: diag_raw = {}
        rank_owner = diag_raw.get('pnp_top1_block')
        if not rank_owner or rank_owner == 'None':
            rank_owner = diag_raw.get('retrieval_top1', 'None')
        def fmt_rank_name(name):
            if not name or name == 'None': return 'None'
            return f"{rank_owner}/{name}"
        ranks = diag_raw.get('db_ranks', [])
        if ranks is None: ranks = []
        ranks = list(ranks)
        while len(ranks) < 3: ranks.append({'name': 'None', 'matches_2d': 0})

        return {
            "Status": diag_raw.get('status', 'Unknown'),
            "PnP_Top1_Block": diag_raw.get('pnp_top1_block', 'None'),
            "PnP_Top1_Inliers": diag_raw.get('pnp_top1_inliers', 0),
            "Retrieval_Top1": diag_raw.get('retrieval_top1', 'None'),
            "R1_Name": fmt_rank_name(ranks[0]['name']),
            "R1_Match": ranks[0]['matches_2d'],
            "R2_Name": fmt_rank_name(ranks[1]['name']),
            "R2_Match": ranks[1]['matches_2d'],
            "R3_Name": fmt_rank_name(ranks[2]['name']),
            "R3_Match": ranks[2]['matches_2d'],
            "Num_Keypoints": diag_raw.get('num_kpts', 0),
            "Num_Matches_3D": diag_raw.get('num_matches_3d', 0),
            "Map_X": diag_raw.get('map_x', ""),
            "Map_Y": diag_raw.get('map_y', ""),
            "Map_Yaw": diag_raw.get('map_yaw', ""),
            
            # --- 以下是更新的詳細時間清單 ---
            "T_Preproc": round(diag_raw.get('time_preprocessing', 0.0), 3),
            "T_Feat_Global_GPU": round(diag_raw.get('time_feat_global_gpu', 0.0), 3),
            "T_Feat_Local_GPU": round(diag_raw.get('time_feat_local_gpu', 0.0), 3),
            "T_Feat_CPU": round(diag_raw.get('time_feat_cpu', 0.0), 3),
            "T_Feat_Total": round(diag_raw.get('time_feat_ext', 0.0), 3),
            "T_Retrieval": round(diag_raw.get('time_global_retrieval', 0.0), 3),
            "T_Match_H5_IO": round(diag_raw.get('time_match_h5_io', 0.0), 3),
            "T_Match_To_GPU": round(diag_raw.get('time_match_transfer_gpu', 0.0), 3),
            "T_Match_LG_GPU": round(diag_raw.get('time_match_lg_gpu', 0.0), 3),
            "T_Match_CPU_Post": round(diag_raw.get('time_match_cpu', 0.0), 3),
            "T_Match_Total": round(diag_raw.get('time_local_matching', 0.0), 3),
            "T_PnP": round(diag_raw.get('time_pnp_transform', 0.0), 3),
            "T_Total": round(diag_raw.get('time_total', 0.0), 3)
        }