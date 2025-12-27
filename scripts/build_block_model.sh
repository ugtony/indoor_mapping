#!/usr/bin/env bash
# build_block_model.sh [V16-Dense-Only]
# - Mode: Pure 360 Dense (8 views) Pipeline
# - Hardcoded: Always uses --dense for 360 conversion
# - Strategy: Directional Window (Axial/Diagonal/Lateral)

set -euox pipefail

# -------- 0. 參數解析 --------
if [ $# -lt 1 ]; then
  echo "Usage: $0 <BLOCK_DATA_DIR> [options]"
  echo "Options:"
  echo "  --fov=FLOAT               Model FOV (Default: 100.0)"
  echo "  --fps=FLOAT               Frame extraction FPS (Default: 2)"
  echo "  --global-conf=STR         Global feature model (megaloc, netvlad...)"
  echo ""
  echo "  [Directional Window Strategy]"
  echo "  --window-axial=INT        Seq window for F/B (Default: 5)"
  echo "  --window-diagonal=INT     Seq window for FR/FL... (Default: -1 Auto)"
  echo "  --window-lateral=INT      Seq window for L/R (Default: 1)"
  echo "  --intra-max-angle=FLOAT   Max intra-frame angle (Default: 90)"
  exit 1
fi

DATA_DIR="$(realpath "$1")"
shift

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "${SCRIPT_DIR}/..")"
CONFIG_FILE="${PROJECT_ROOT}/project_config.env"

# 1. 預設值
DEFAULT_FOV="AUTO"
DEFAULT_GLOBAL="megaloc"
DEFAULT_FPS=2

# 策略預設值
DEFAULT_WIN_AXIAL=5
DEFAULT_WIN_DIAGONAL=-1
DEFAULT_WIN_LATERAL=1
DEFAULT_INTRA_ANGLE=90.0
DEFAULT_CROSS_OVER=1

# 2. 載入設定檔
if [ -f "${CONFIG_FILE}" ]; then
  echo "[Init] Loading config from ${CONFIG_FILE}..."
  source "${CONFIG_FILE}"
fi

# 3. 變數初始化 (CLI > Config > Default)
FOV_MODEL_VAL="${FOV_MODEL:-$DEFAULT_FOV}"
GLOBAL_CONF="${GLOBAL_CONF:-$DEFAULT_GLOBAL}"
EXTRACT_FPS="${FPS:-$DEFAULT_FPS}"

WIN_AXIAL="${WINDOW_AXIAL:-$DEFAULT_WIN_AXIAL}"
WIN_DIAGONAL="${WINDOW_DIAGONAL:-$DEFAULT_WIN_DIAGONAL}"
WIN_LATERAL="${WINDOW_LATERAL:-$DEFAULT_WIN_LATERAL}"
INTRA_ANGLE="${INTRA_MAX_ANGLE:-$DEFAULT_INTRA_ANGLE}"
CROSS_OVER="${ENABLE_CROSS_OVER:-$DEFAULT_CROSS_OVER}"

# 4. 解析 CLI 參數
while [ $# -gt 0 ]; do
  case "$1" in
    --fov=*)  FOV_MODEL_VAL="${1#*=}" ;;
    --fps=*)  EXTRACT_FPS="${1#*=}" ;;
    
    # Directional Strategy
    --window-axial=*)    WIN_AXIAL="${1#*=}" ;;
    --window-diagonal=*) WIN_DIAGONAL="${1#*=}" ;;
    --window-lateral=*)  WIN_LATERAL="${1#*=}" ;;
    --intra-max-angle=*) INTRA_ANGLE="${1#*=}" ;;
    --enable-cross-over=*) CROSS_OVER="${1#*=}" ;;

    --global-conf=*|--global_model=*) GLOBAL_CONF="${1#*=}" ;;
    *) ;;
  esac
  shift
done

# 智慧 FOV 邏輯 (Dense 專用)
if [ "${FOV_MODEL_VAL}" = "AUTO" ]; then
  FOV_MODEL_VAL=100.0 # Dense 模式的最佳實踐
fi

# -------- 1. 路徑與環境設定 --------
BLOCK_NAME="$(basename "${DATA_DIR}")"
OUT_ROOT="${PROJECT_ROOT}/outputs-hloc"
OUT_DIR="${OUT_ROOT}/${BLOCK_NAME}"
LOG_DIR="${OUT_DIR}/logs"
VIZ_DIR="${OUT_DIR}/visualization"
DBG_DIR="${OUT_DIR}/debug"

mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${VIZ_DIR}" "${DBG_DIR}"

# 備份設定檔
[ -f "${CONFIG_FILE}" ] && cp "${CONFIG_FILE}" "${OUT_DIR}/config_used.env"

if [ -x "/opt/conda/bin/python" ]; then PY="/opt/conda/bin/python"; else PY="${PY:-python3}"; fi

echo "========================================"
echo "[Info] Block: ${BLOCK_NAME}"
echo "[Info] Pipeline: 360 Dense (8 Views)"
echo "[Info] Model FOV: ${FOV_MODEL_VAL}"
echo "[Info] Global Model: ${GLOBAL_CONF}"
echo "[Info] Strategy: Axial=${WIN_AXIAL}, Diag=${WIN_DIAGONAL}, Lat=${WIN_LATERAL}"
echo "[Info] Intra-Angle: ${INTRA_ANGLE}°"
echo "[Info] Output: ${OUT_DIR}"
echo "========================================"

# -------- 2. 組態設定 --------
LOCAL_CONF="superpoint_aachen"
MATCHER_CONF="superpoint+lightglue"
REBUILD_SFM="${REBUILD_SFM:-0}"
ALIGN_SFM="${ALIGN_SFM:-1}"

# -------- 3. 核心檔案路徑 --------
DB_LIST="${OUT_DIR}/db.txt"
LOCAL_FEATS="${OUT_DIR}/local-${LOCAL_CONF}.h5"
GLOBAL_FEATS="${OUT_DIR}/global-${GLOBAL_CONF}.h5"
PAIRS_DB="${OUT_DIR}/pairs-raw.txt"
PAIRS_DB_CLEAN="${OUT_DIR}/pairs-clean.txt"
DB_MATCHES="${OUT_DIR}/db-matches-${MATCHER_CONF}.h5"
SFM_DIR="${OUT_DIR}/sfm"
SFM_ALIGNED="${OUT_DIR}/sfm_aligned"
STAGE="${OUT_DIR}/_images_stage"

# Scripts
ALIGN_SCRIPT="${PROJECT_ROOT}/scripts/align_linear_path.py"
VIZ_SCRIPT="${PROJECT_ROOT}/scripts/visualize_sfm_open3d.py"
CONVERT_360_SCRIPT="${PROJECT_ROOT}/scripts/convert360_to_pinhole.py"
PAIRS_360_SCRIPT="${PROJECT_ROOT}/scripts/pairs_from_360.py"
EXTRACT_SCRIPT="${PROJECT_ROOT}/scripts/extract_frames.sh"

# -------- [Step -1] 自動抽幀 --------
RAW_DIR="${DATA_DIR}/raw"
TARGET_EXT_DIR="${DATA_DIR}/db_360"
DST_DB="${DATA_DIR}/db"

VIDEO_EXTS=(-iname "*.mp4" -o -iname "*.mov" -o -iname "*.insv" -o -iname "*.360" -o -iname "*.mkv")

if [ -d "${RAW_DIR}" ]; then
  HAS_VIDEO=$(find "${RAW_DIR}" -maxdepth 1 -type f \( "${VIDEO_EXTS[@]}" \) -print -quit)
  if [ -n "$HAS_VIDEO" ]; then
      HAS_IMAGES=""
      if [ -d "${TARGET_EXT_DIR}" ]; then
          HAS_IMAGES=$(find "${TARGET_EXT_DIR}" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.png" \) -print -quit)
      fi
      if [ -n "$HAS_IMAGES" ]; then
          echo "[-1] Found existing images in ${TARGET_EXT_DIR}. Skipping extraction."
      else
          echo "[-1] Found video in raw/. Extracting to db_360/ (FPS=${EXTRACT_FPS})..."
          bash "${EXTRACT_SCRIPT}" "${RAW_DIR}" "${TARGET_EXT_DIR}" \
              --fps "${EXTRACT_FPS}" --prefix "frames" --ext "jpg"
      fi
  fi
fi

# -------- [Step 0] 360 Pinhole 轉換 (Force Dense) --------
echo "[0] 360 Pinhole Conversion (8 views)..."
HAS_DB_IMAGES=""
if [ -d "${DST_DB}" ]; then
    HAS_DB_IMAGES=$(find "${DST_DB}" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.png" \) -print -quit)
fi

if [ -n "$HAS_DB_IMAGES" ]; then
    echo "    > Target DB ${DST_DB} is not empty. Skipping conversion."
else
    echo "    > Converting Equirectangular to Pinhole..."
    if [ ! -d "${TARGET_EXT_DIR}" ]; then echo "[Error] Missing ${TARGET_EXT_DIR}"; exit 1; fi
    # Force --dense argument here
    CONVERT_ARGS=( "--input_dir" "${TARGET_EXT_DIR}" "--output_dir" "${DST_DB}" "--fov" "${FOV_MODEL_VAL}" "--dense" )
    "${PY}" "${CONVERT_360_SCRIPT}" "${CONVERT_ARGS[@]}"
fi

# -------- HLOC Pipeline --------
echo "[1] Generating DB image list..."
if [ ! -d "${DST_DB}" ]; then echo "[Error] ${DST_DB} not found."; exit 1; fi
(cd "${DATA_DIR}" && find db -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.png' \) | sort) > "${DB_LIST}"
if [ ! -s "${DB_LIST}" ]; then echo "[Error] No images in db/."; exit 1; fi

echo "[2] Checking integrity of local features H5..."
${PY} - <<PY || { echo "[Check] H5 stale/corrupted. Deleting."; rm -f "${LOCAL_FEATS}"; }
import h5py, sys; from pathlib import Path
db_paths=[l.strip() for l in Path("${DB_LIST}").read_text().splitlines() if l.strip()]
ok=True
try:
    with h5py.File("${LOCAL_FEATS}","r") as f:
        for p in db_paths:
            if p not in f: ok=False; break
except Exception: ok=False
sys.exit(0 if ok else 1)
PY

echo "[3] Extracting LOCAL features (${LOCAL_CONF})..."
${PY} -m hloc.extract_features --conf "${LOCAL_CONF}" \
  --image_dir "${DATA_DIR}" --image_list "${DB_LIST}" \
  --export_dir "${OUT_DIR}" --feature_path "${LOCAL_FEATS}"

echo "[4] Extracting GLOBAL features (${GLOBAL_CONF})..."
${PY} -m hloc.extract_features --conf "${GLOBAL_CONF}" \
  --image_dir "${DATA_DIR}" --image_list "${DB_LIST}" \
  --export_dir "${OUT_DIR}" --feature_path "${GLOBAL_FEATS}"

echo "[5] Building DB pairs (Directional Strategy)..."
PAIRS_ARGS=( \
  "--db_list" "${DB_LIST}" \
  "--output" "${PAIRS_DB}" \
  "--window_axial" "${WIN_AXIAL}" \
  "--window_diagonal" "${WIN_DIAGONAL}" \
  "--window_lateral" "${WIN_LATERAL}" \
  "--intra_max_angle" "${INTRA_ANGLE}" \
)

if [ "${CROSS_OVER}" = "1" ]; then
    PAIRS_ARGS+=( "--enable_cross_over" )
fi

"${PY}" "${PAIRS_360_SCRIPT}" "${PAIRS_ARGS[@]}"

echo "[6] Cleaning pairs list..."
${PY} - <<PY
from pathlib import Path; import h5py, sys
pairs_in = Path("${PAIRS_DB}"); pairs_out = Path("${PAIRS_DB_CLEAN}")
db_list = set([l.strip() for l in Path("${DB_LIST}").read_text().splitlines() if l.strip()])
try:
    with h5py.File("${LOCAL_FEATS}","r") as f, open(pairs_in,"r") as fi, open(pairs_out,"w") as fo:
        keep=0
        for line in fi:
            s=line.strip().split()
            if len(s)<2: continue
            a,b = s[0], s[1]
            if (a in db_list) and (b in db_list) and (a in f) and (b in f):
                fo.write(line); keep+=1
        print(f"    > Pairs cleaned: {keep} kept.")
except Exception as e: print(f"[Error] Clean failed: {e}", file=sys.stderr); sys.exit(1)
PY
PAIRS_USE="${PAIRS_DB_CLEAN}"

echo "[7] Matching DB pairs (${MATCHER_CONF})..."
${PY} - <<PY
from pathlib import Path
from hloc import match_features
match_features.main(
    conf=match_features.confs["${MATCHER_CONF}"],
    pairs=Path("${PAIRS_USE}"),
    features=Path("${LOCAL_FEATS}"),
    matches=Path("${DB_MATCHES}")
)
PY

echo "[8] Running SfM reconstruction..."
if [ "${REBUILD_SFM}" = "1" ]; then rm -rf "${SFM_DIR}"; fi
if [ -f "${SFM_DIR}/images.bin" ]; then
  echo "    > SfM exists. Skipping."
else
  echo "    > [Fix] Staging images from db.txt to clean directory..."
  rm -rf "${STAGE}"; mkdir -p "${STAGE}"
  set +x
  while read -r rel_path; do
      [ -z "$rel_path" ] && continue
      SRC_FILE="${DATA_DIR}/${rel_path}"; DST_FILE="${STAGE}/${rel_path}"
      mkdir -p "$(dirname "${DST_FILE}")"
      ln "${SRC_FILE}" "${DST_FILE}" 2>/dev/null || cp "${SRC_FILE}" "${DST_FILE}"
  done < "${DB_LIST}"
  set -x
  
  ${PY} -m hloc.reconstruction \
    --image_dir "${STAGE}" --pairs "${PAIRS_USE}" \
    --features "${LOCAL_FEATS}" --matches "${DB_MATCHES}" \
    --sfm_dir "${SFM_DIR}" | tee "${LOG_DIR}/reconstruction.log"
fi

echo "[9] Verifying SfM model..."
NEED_SWAP=false; BEST_MODEL_PATH=""; MAX_IMG_COUNT=0; ROOT_IMG_COUNT=0
if [ -f "${SFM_DIR}/images.bin" ]; then
    ROOT_IMG_COUNT=$(${PY} -c "import pycolmap, sys; print(len(pycolmap.Reconstruction(sys.argv[1]).images))" "${SFM_DIR}" 2>/dev/null || echo 0)
fi
if [ -d "${SFM_DIR}/models" ]; then
  for MODEL_DIR in "${SFM_DIR}/models"/*; do
    if [ -d "${MODEL_DIR}" ] && [ -f "${MODEL_DIR}/images.bin" ]; then
      CNT=$(${PY} -c "import pycolmap, sys; print(len(pycolmap.Reconstruction(sys.argv[1]).images))" "${MODEL_DIR}" 2>/dev/null || echo 0)
      if [ "${CNT}" -gt "${MAX_IMG_COUNT}" ]; then MAX_IMG_COUNT="${CNT}"; BEST_MODEL_PATH="${MODEL_DIR}"; fi
    fi
  done
fi
if [ "${ROOT_IMG_COUNT}" -lt "${MAX_IMG_COUNT}" ] && [ -n "${BEST_MODEL_PATH}" ]; then
  echo "    [Fix] Swapping root (${ROOT_IMG_COUNT}) with best (${MAX_IMG_COUNT}) from ${BEST_MODEL_PATH##*/}..."
  TMP="${SFM_DIR}/_swap"; mkdir -p "${TMP}"
  find "${SFM_DIR}" -maxdepth 1 -type f -name "*.bin" -exec mv {} "${TMP}/" \;
  find "${BEST_MODEL_PATH}" -maxdepth 1 -type f -name "*.bin" -exec mv {} "${SFM_DIR}/" \;
  find "${TMP}" -maxdepth 1 -type f -name "*.bin" -exec mv {} "${BEST_MODEL_PATH}/" \;
  rm -rf "${TMP}"
else
  echo "    > Selection OK (Root: ${ROOT_IMG_COUNT}, Max: ${MAX_IMG_COUNT})."
fi

echo "[10] Align to Linear Path (Normalized)"
if [ "${ALIGN_SFM}" = "1" ] && [ -d "${SFM_DIR}" ]; then
  "${PY}" "${ALIGN_SCRIPT}" "${SFM_DIR}" "${SFM_ALIGNED}" > "${LOG_DIR}/align.log" 2>&1
  FINAL_SFM="${SFM_ALIGNED}"
else
  FINAL_SFM="${SFM_DIR}"
fi
if [ -f "${VIZ_SCRIPT}" ] && [ -d "${FINAL_SFM}" ]; then
  "${PY}" "${VIZ_SCRIPT}" --sfm_dir "${FINAL_SFM}" --output_dir "${VIZ_DIR}" --no_server >/dev/null
fi

echo "✅ Completed: ${BLOCK_NAME} (Dense 360)"