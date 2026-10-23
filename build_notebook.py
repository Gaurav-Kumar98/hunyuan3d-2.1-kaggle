"""Builds the Kaggle notebooks. Run: python build_notebook.py <output_dir> [scripts_dir]"""
import json
import sys

INSTALL_SH = r'''#!/usr/bin/env bash
# Installs Hunyuan3D 2.1 on top of Kaggle's preinstalled PyTorch (no torch reinstall).
set -euo pipefail
PY=${PY:-python}
REPO=${REPO:-/tmp/Hunyuan3D-2.1}
BUILD_CUDA=${BUILD_CUDA:-1}
mkdir -p /tmp/hy3d

[ -d "$REPO" ] || git clone -q --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git "$REPO"
cd "$REPO"
"$PY" -c "import sys, torch; print('python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.version.cuda)"

# numpy 1.26 has no wheels for Python 3.13+, where numpy 2.2 is tested to work.
"$PY" -c "import sys; print('numpy==1.26.4' if sys.version_info < (3, 13) else 'numpy<2.3')" > /tmp/hy3d/constraints.txt
PIP="$PY -m pip install -q --root-user-action=ignore -c /tmp/hy3d/constraints.txt"

echo "Installing Python packages..."
$PIP numpy \
  diffusers==0.30.0 transformers==4.46.0 accelerate==1.1.1 huggingface-hub==0.30.2 hf_transfer \
  pytorch-lightning==1.9.5 safetensors einops==0.8.0 omegaconf==2.3.0 pyyaml tqdm psutil configargparse \
  opencv-python==4.10.0.84 opencv-python-headless==4.10.0.84 scikit-image imageio \
  trimesh==4.4.7 "pymeshlab>=2022.2.post3" pygltflib==1.16.3 xatlas \
  rembg==2.0.65 onnxruntime timm torchdiffeq ninja pybind11 spandrel

# bpy (Blender) has no wheel for this Python and is only used by an OBJ->GLB helper we replace.
sed -i 's/^import bpy$/try:\n    import bpy\nexcept ImportError:\n    bpy = None/' hy3dpaint/DifferentiableRenderer/mesh_utils.py
# basicsr/realesrgan (the texture upscaler) fail to install on Python 3.13; swap in spandrel.
cat > hy3dpaint/utils/image_super_utils.py <<'PYEOF'
__SUPER_PY__PYEOF

EXT=$("$PY" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
echo "Building mesh inpainter..."
(cd hy3dpaint/DifferentiableRenderer && \
  c++ -O3 -shared -std=c++11 -fPIC $("$PY" -m pybind11 --includes) mesh_inpaint_processor.cpp -o "mesh_inpaint_processor$EXT")

if [ "$BUILD_CUDA" = 1 ]; then
  export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
  export PATH="$CUDA_HOME/bin:$PATH"
  command -v nvcc >/dev/null || { echo "nvcc not found in $CUDA_HOME/bin - is a GPU accelerator enabled?"; exit 1; }
  # Compile only for the GPUs in this session (T4 = 7.5, P100 = 6.0): much faster than all archs.
  export TORCH_CUDA_ARCH_LIST=$("$PY" -c "import torch; print(';'.join(sorted({'%d.%d' % torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())})))")
  echo "Building custom rasterizer for CUDA arch $TORCH_CUDA_ARCH_LIST (takes a few minutes)..."
  (cd hy3dpaint/custom_rasterizer && $PY -m pip install -q --root-user-action=ignore --no-build-isolation .)
fi

echo "Checking imports..."
"$PY" - <<'EOF'
import os, sys
sys.path[:0] = ["hy3dshape", "hy3dpaint"]
from DifferentiableRenderer import mesh_inpaint_processor
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
if os.environ.get("BUILD_CUDA", "1") == "1":
    import custom_rasterizer_kernel
    from textureGenPipeline import Hunyuan3DPaintPipeline
from convert_utils import create_glb_with_pbr_materials
import numpy
print("Install OK (numpy %s)" % numpy.__version__)
EOF
'''

SUPER_PY = r'''# Replaces the basicsr/realesrgan upscaler: neither package installs on Python 3.13+.
# spandrel loads the same RealESRGAN_x4plus.pth weights.
import numpy as np
import torch
from PIL import Image
from spandrel import ImageModelDescriptor, ModelLoader


class imageSuperNet:
    def __init__(self, config) -> None:
        model = ModelLoader().load_from_file(config.realesrgan_ckpt_path)
        assert isinstance(model, ImageModelDescriptor)
        device = torch.device(config.device)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.model = model.to(device).to(dtype).eval()
        self.device, self.dtype = device, dtype

    @torch.no_grad()
    def __call__(self, image):
        x = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1)[None]
        y = self.model(x.to(self.device, self.dtype) / 255.0).clamp(0, 1)
        return Image.fromarray((y[0].permute(1, 2, 0).float().cpu().numpy() * 255).round().astype(np.uint8))
'''

PREFETCH_PY = r'''"""Downloads every weight file (~17.8 GB) into one folder, in parallel.

Usage: python prefetch.py <weights_dir>

Layout (the workers and the Kaggle weights dataset both use it):
  <weights_dir>/tencent/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/       shape model      6.9 GB
  <weights_dir>/tencent/Hunyuan3D-2.1/hunyuan3d-paintpbr-v2-1/  texture model    6.4 GB
  <weights_dir>/facebook/dinov2-giant/                          texture encoder  4.2 GB
  <weights_dir>/RealESRGAN_x4plus.pth                           upscaler
  <weights_dir>/u2net/u2net.onnx                                background removal
"""
import os, shutil, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

try:
    import hf_transfer  # noqa: F401  (several times faster on Kaggle)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
except ImportError:
    pass
from huggingface_hub import snapshot_download

W = sys.argv[1]
DIT = "tencent/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/model.fp16.ckpt"
PAINT = "tencent/Hunyuan3D-2.1/hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.bin"
DINO = "facebook/dinov2-giant/model.safetensors"
ESRGAN = "RealESRGAN_x4plus.pth"
U2NET = "u2net/u2net.onnx"


def have(rel):
    return os.path.exists(os.path.join(W, rel))


def hub(repo_id, patterns):
    # One call per repo: parallel calls sharing a local_dir clobber each other's temp files.
    if patterns:
        snapshot_download(repo_id=repo_id, allow_patterns=patterns, local_dir=os.path.join(W, repo_id))
    return repo_id


def url(link, rel):
    dest = os.path.join(W, rel)
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        urllib.request.urlretrieve(link, dest + ".part")
        os.replace(dest + ".part", dest)
    return rel


# Anything already present (e.g. linked in from a partial dataset) is skipped.
tasks = [
    lambda: hub("tencent/Hunyuan3D-2.1", ([] if have(DIT) else ["hunyuan3d-dit-v2-1/*"])
                + ([] if have(PAINT) else ["hunyuan3d-paintpbr-v2-1/*"])),
    lambda: hub("facebook/dinov2-giant", [] if have(DINO) else ["*.json", "*.safetensors", "*.txt"]),
    lambda: url("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth", ESRGAN),
    lambda: url("https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx", U2NET),
]
with ThreadPoolExecutor(len(tasks)) as pool:
    for done in pool.map(lambda task: task(), tasks):
        print("ready:", done)

for repo_id in ("tencent/Hunyuan3D-2.1", "facebook/dinov2-giant"):  # download bookkeeping only
    shutil.rmtree(os.path.join(W, repo_id, ".cache"), ignore_errors=True)

missing = [rel for rel in (DIT, PAINT, DINO, ESRGAN, U2NET) if not have(rel)]
if missing:
    sys.exit(f"Download incomplete, missing: {missing}")
size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(W) for f in fs)
print(f"All weights present: {size / 2**30:.1f} GB in {W}")
'''

STAGE_PY = r'''"""Copies the weights dataset to local disk with many parallel reads.

Kaggle streams dataset files over the network on first read (~17 MB/s for one sequential reader), so the models
loaded straight from /kaggle/input in 7+ min each. Reading 64 MB chunks on 16 threads runs during the install.
Usage: python stage_weights.py <dataset_dir> <local_dir>
"""
import os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

src, dst = sys.argv[1], sys.argv[2]
CHUNK = 64 << 20

files = [os.path.join(d, f) for d, _, fs in os.walk(src) for f in fs]
total = sum(os.path.getsize(p) for p in files)
jobs = []
for p in files:
    out = os.path.join(dst, os.path.relpath(p, src))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    size = os.path.getsize(p)
    with open(out, "wb") as f:
        f.truncate(size)
    jobs += [(p, out, off, min(CHUNK, size - off)) for off in range(0, size, CHUNK)]
jobs.sort(key=lambda j: j[2])  # spread early chunks across all files

copied, lock = 0, threading.Lock()


def copy(job):
    global copied
    p, out, off, n = job
    with open(p, "rb") as fi, open(out, "r+b") as fo:
        fi.seek(off)
        fo.seek(off)
        while n:
            buf = fi.read(min(n, 8 << 20))
            if not buf:
                raise IOError(f"short read: {p}")
            fo.write(buf)
            n -= len(buf)
            with lock:
                copied += len(buf)


def report():
    while copied < total:
        time.sleep(30)
        print(f"staged {copied / 2**30:.1f} / {total / 2**30:.1f} GB "
              f"({copied / 2**20 / (time.time() - start):.0f} MB/s)", flush=True)


start = time.time()
threading.Thread(target=report, daemon=True).start()
with ThreadPoolExecutor(16) as pool:
    list(pool.map(copy, jobs))
open(os.path.join(dst, "_STAGED"), "w").close()
print(f"STAGED {total / 2**30:.1f} GB in {time.time() - start:.0f}s", flush=True)
'''

SHAPE_WORKER_PY = r'''"""GPU worker 1: image -> untextured mesh. Each finished mesh is queued for paint_worker.py."""
import json, os, sys, time, traceback

cfg = json.load(open(sys.argv[1]))
os.chdir(cfg["repo"])
sys.path.insert(0, os.path.join(cfg["repo"], "hy3dshape"))

import torch
from PIL import Image
from hy3dshape import FaceReducer
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.rembg import BackgroundRemover


def log(msg):
    print(f"[shape {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def enqueue(job):
    tmp = os.path.join(cfg["queue"], f".{job['name']}.tmp")
    with open(tmp, "w") as f:
        json.dump(job, f)
    os.replace(tmp, os.path.join(cfg["queue"], f"{job['name']}.json"))  # atomic: paint never sees half a file


import faulthandler
faulthandler.dump_traceback_later(300, repeat=True)  # while loading: log the stack every 5 min, so a stall shows where
t = time.time()
pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    "tencent/Hunyuan3D-2.1", subfolder="hunyuan3d-dit-v2-1", use_safetensors=False, device="cuda")
rembg = BackgroundRemover()
face_reducer = FaceReducer()
faulthandler.cancel_dump_traceback_later()
log(f"model loaded on {torch.cuda.get_device_name(0)} in {time.time() - t:.0f}s")
open(os.path.join(cfg["queue"], "_SHAPE_READY"), "w").close()

for item in cfg["images"]:
    name, out = item["name"], os.path.join(cfg["out"], item["name"])
    os.makedirs(out, exist_ok=True)
    try:
        if cfg["skip_existing"] and os.path.exists(os.path.join(out, "white_mesh.obj")):
            log(f"{name}: mesh already exists, skipping")
            enqueue({"name": name, "dir": out})
            continue
        t = time.time()
        torch.cuda.reset_peak_memory_stats()
        image = Image.open(item["path"])
        if image.mode != "RGBA" or cfg["force_rembg"]:
            image = rembg(image.convert("RGB"))
        image.save(os.path.join(out, "input.png"))

        mesh = pipe(
            image=image,
            num_inference_steps=cfg["steps"],
            guidance_scale=cfg["guidance_scale"],
            generator=torch.Generator().manual_seed(cfg["seed"]),
            octree_resolution=cfg["octree_resolution"],
            num_chunks=cfg["num_chunks"],
            output_type="trimesh",
        )[0]
        mesh = face_reducer(mesh, max_facenum=cfg["max_faces"])
        mesh.export(os.path.join(out, "white_mesh.glb"))
        mesh.export(os.path.join(out, "white_mesh.obj"), include_normals=False)
        log(f"{name}: {len(mesh.faces)} faces in {time.time() - t:.0f}s "
            f"(peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GB)")
        enqueue({"name": name, "dir": out})
    except Exception:
        log(f"{name}: FAILED\n{traceback.format_exc()}")
    finally:
        torch.cuda.empty_cache()

open(os.path.join(cfg["queue"], "_SHAPE_DONE"), "w").close()
log("all shapes done")
'''

PAINT_WORKER_PY = r'''"""GPU worker 2: untextured mesh + reference image -> PBR-textured GLB, fed by shape_worker.py."""
import glob, json, os, sys, time, traceback

cfg = json.load(open(sys.argv[1]))
os.chdir(cfg["repo"])
sys.path[:0] = [cfg["repo"], os.path.join(cfg["repo"], "hy3dpaint")]

import base64
import pygltflib
import pymeshlab
import torch
import textureGenPipeline
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from convert_utils import create_glb_with_pbr_materials


def log(msg):
    print(f"[paint {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def remesh_to_max_faces(mesh_path, remesh_path):
    """Replaces the repo's remesh_mesh, which cuts every mesh to 40k faces (with open3d, which isn't installed).

    That cap silently threw away the detail a higher max_faces setting asked for; this keeps up to max_faces.
    """
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_path)
    if ms.current_mesh().face_number() > cfg["max_faces"]:
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=cfg["max_faces"])
    ms.save_current_mesh(remesh_path, save_textures=False)


textureGenPipeline.remesh_mesh = remesh_to_max_faces  # the pipeline looks it up as a module global at call time


def embed_textures(path):
    """Moves textures into the GLB binary chunk with the right mimeType.

    create_glb_with_pbr_materials stores them as JSON data URIs all labelled image/png (the base colour is a JPEG).
    Web viewers accept that, but it is invalid GLB: Windows 3D Viewer and stricter importers show no texture.
    """
    gltf = pygltflib.GLTF2().load(path)
    blob = bytearray(gltf.binary_blob() or b"")
    changed = False
    for image in gltf.images:
        if not (image.uri or "").startswith("data:"):
            continue
        data = base64.b64decode(image.uri.split(",", 1)[1])
        blob += b"\x00" * (-len(blob) % 4)  # bufferViews must be 4-byte aligned
        gltf.bufferViews.append(pygltflib.BufferView(buffer=0, byteOffset=len(blob), byteLength=len(data)))
        blob += data
        mime = "image/jpeg" if data[:3] == b"\xff\xd8\xff" else "image/png"
        image.uri, image.bufferView, image.mimeType = None, len(gltf.bufferViews) - 1, mime
        changed = True
    if not changed:
        return
    for material in gltf.materials:
        if material.alphaMode in (None, "OPAQUE"):
            material.alphaCutoff = None  # only meaningful with alphaMode MASK
    blob += b"\x00" * (-len(blob) % 4)
    gltf.buffers[0].byteLength = len(blob)
    gltf.set_binary_blob(bytes(blob))
    gltf.save_binary(path)


def next_job():
    jobs = sorted(glob.glob(os.path.join(cfg["queue"], "*.json")), key=os.path.getmtime)
    return jobs[0] if jobs else None


# The texture model always calls snapshot_download(); serve it from the weights folder instead.
import huggingface_hub
_download = huggingface_hub.snapshot_download
def _local_first(repo_id, *args, **kwargs):
    local = os.path.join(cfg["weights"], repo_id)
    return local if os.path.isdir(local) else _download(repo_id, *args, **kwargs)
huggingface_hub.snapshot_download = _local_first  # multiview_utils looks it up at call time

import faulthandler
faulthandler.dump_traceback_later(300, repeat=True)  # while loading: log the stack every 5 min, so a stall shows where
t = time.time()
conf = Hunyuan3DPaintConfig(cfg["texture_views"], cfg["texture_resolution"])
conf.dino_ckpt_path = os.path.join(cfg["weights"], "facebook/dinov2-giant")
conf.realesrgan_ckpt_path = os.path.join(cfg["weights"], "RealESRGAN_x4plus.pth")
conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
pipe = Hunyuan3DPaintPipeline(conf)
faulthandler.cancel_dump_traceback_later()
log(f"model loaded on {torch.cuda.get_device_name(0)} in {time.time() - t:.0f}s")

while True:
    job_file = next_job()
    if job_file is None:
        # Shape writes _SHAPE_DONE after its last enqueue, so re-check the queue once it appears.
        if os.path.exists(os.path.join(cfg["queue"], "_SHAPE_DONE")) and next_job() is None:
            break
        time.sleep(2)
        continue

    job = json.load(open(job_file))
    os.remove(job_file)
    name, out = job["name"], job["dir"]
    glb_path = os.path.join(out, "textured_mesh.glb")
    if cfg["skip_existing"] and os.path.exists(glb_path):
        embed_textures(glb_path)  # also repairs GLBs written before this fix
        log(f"{name}: textured mesh already exists, skipping")
        continue
    try:
        t = time.time()
        torch.cuda.reset_peak_memory_stats()
        obj_path = pipe(
            mesh_path=os.path.join(out, "white_mesh.obj"),
            image_path=os.path.join(out, "input.png"),
            output_mesh_path=os.path.join(out, "textured_mesh.obj"),
            save_glb=False,
        )
        base = obj_path[:-len(".obj")]
        create_glb_with_pbr_materials(
            obj_path,
            {"albedo": base + ".jpg", "metallic": base + "_metallic.jpg", "roughness": base + "_roughness.jpg"},
            glb_path,
        )
        embed_textures(glb_path)
        log(f"{name}: textured in {time.time() - t:.0f}s "
            f"(peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GB)")
    except torch.cuda.OutOfMemoryError:
        log(f"{name}: OUT OF GPU MEMORY - lower texture_views / texture_resolution in the settings cell")
    except Exception:
        log(f"{name}: FAILED\n{traceback.format_exc()}")
    finally:
        torch.cuda.empty_cache()

log("all textures done")
'''


INSTALL_SH = INSTALL_SH.replace("__SUPER_PY__", SUPER_PY.lstrip("\n"))


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")}


def writefile(path, body):
    return code(f"%%writefile {path}\n{body.strip(chr(10))}")


cells = [
    md("""
# Hunyuan3D 2.1 on Kaggle, using both T4 GPUs

Turns images into textured 3D models (`.glb`) with [Tencent Hunyuan3D 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1).

**How the two GPUs are used:** the model has two stages, so each stage gets its own T4.

| GPU | Worker | Does |
|---|---|---|
| `cuda:0` | `shape_worker.py` | removes background, then image to untextured mesh (~3.3B-param model) |
| `cuda:1` | `paint_worker.py` | untextured mesh to PBR texture (multiview diffusion + upscaler + baking) |

The two run as separate processes linked by a small folder queue. While GPU 1 is texturing model *N*, GPU 0 is already
building model *N+1*. Neither stage has to share 16 GB with the other, and with a batch of images, throughput roughly doubles.
With only one GPU, the notebook runs the stages one after the other on that GPU.

### Before you run
1. **Settings → Accelerator → GPU T4 x2**
2. **Settings → Internet → On** (needs a phone-verified Kaggle account)
3. Optional: **Add Input → Upload** a folder of images (PNG with a transparent background works best; other images get their background removed automatically).
4. Recommended: **Add Input → Your Datasets →** your Hunyuan3D weights dataset, so the 17.8 GB of weights don't download every session. Build it once with `hunyuan3d_2_1_weights_builder.ipynb`.
5. **Run All.** Setup takes about 10 minutes (package installs and compiling the CUDA rasterizer), plus 5–10 minutes more if it has to download the weights.

Results go to `/kaggle/working/hy3d_out/<image name>/textured_mesh.glb` and get zipped into `/kaggle/working/hy3d_results.zip`.
For long batches, use **Save Version → Save & Run All (Commit)** so it keeps running after you close the tab.

> License: the Tencent Hunyuan 3D 2.1 Community License applies. Read it before commercial use. It doesn't cover the EU, UK or South Korea.
"""),
    md("""
## 1. Check the GPUs and start copying weights to local disk
If a weights dataset is attached, it is copied to `/tmp/hy3d-weights` in the background while the install runs.
Loading straight from `/kaggle/input` is slow (~7 min per model), because Kaggle streams dataset files over the network.
"""),
    writefile("/tmp/stage_weights.py", STAGE_PY),
    code(r'''
import glob, os, shutil, subprocess, sys
!nvidia-smi --query-gpu=index,name,memory.total --format=csv

WEIGHTS = "/tmp/hy3d-weights"
found = glob.glob("/kaggle/input/**/tencent/Hunyuan3D-2.1", recursive=True)
DATASET = found[0].split("/tencent/")[0] if found else None
stage_proc = None
if DATASET and not os.path.exists(f"{WEIGHTS}/_STAGED"):
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(DATASET) for f in fs)
    free = shutil.disk_usage("/tmp").free
    if free > size + (5 << 30):
        print(f"Copying {size / 2**30:.1f} GB from {DATASET} to {WEIGHTS} in the background...")
        stage_proc = subprocess.Popen([sys.executable, "/tmp/stage_weights.py", DATASET, WEIGHTS],
                                      stdout=open("/tmp/stage_weights.log", "w"), stderr=subprocess.STDOUT)
    else:
        print(f"Only {free / 2**30:.0f} GB free in /tmp - models will load from the dataset directly (slower).")
        WEIGHTS = DATASET
elif not DATASET:
    print("No weights dataset attached - weights will be downloaded in step 3.")
'''),
    md("## 2. Install (about 10 minutes)\nClones the repo to `/tmp`, installs pinned dependencies on top of Kaggle's PyTorch, and compiles the CUDA rasterizer and mesh inpainter for the GPUs in this session."),
    code("!mkdir -p /tmp/hy3d"),
    writefile("/tmp/hy3d/install.sh", INSTALL_SH),
    code("""
import os, subprocess, sys
# subprocess (not !bash) so a failed install stops Run All instead of scrolling past.
subprocess.run(["bash", "/tmp/hy3d/install.sh"], env=dict(os.environ, PY=sys.executable), check=True)
"""),
    md("""
## 3. Weights: attached dataset, or download (17.8 GB)
If you attached the weights dataset (built once with `hunyuan3d_2_1_weights_builder.ipynb`), this waits for the
background copy started in step 1 and downloads nothing. Without one, it downloads everything to `/tmp`,
which takes about 5–10 minutes per session.
"""),
    writefile("/tmp/hy3d/prefetch.py", PREFETCH_PY),
    code(r'''
import os, subprocess, sys, time

REQUIRED = ["tencent/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/model.fp16.ckpt",
            "tencent/Hunyuan3D-2.1/hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.bin",
            "facebook/dinov2-giant/model.safetensors",
            "RealESRGAN_x4plus.pth",
            "u2net/u2net.onnx"]

def stage_status():
    lines = open("/tmp/stage_weights.log").read().strip().splitlines()
    return lines[-1] if lines else "copying weights to local disk..."

if stage_proc is not None:  # started in step 1; usually done by now
    while stage_proc.poll() is None:
        print(stage_status())
        time.sleep(30)
    print(stage_status())
    if stage_proc.returncode != 0:
        print("Copy failed - loading from the dataset directly instead.")
        WEIGHTS = DATASET

missing = [f for f in REQUIRED if not os.path.exists(f"{WEIGHTS}/{f}")]
if missing:
    if WEIGHTS == DATASET:  # never write into the read-only dataset
        WEIGHTS = "/tmp/hy3d-weights"
        subprocess.run(["cp", "-rsf", f"{DATASET}/.", WEIGHTS], check=True)  # symlinks, no copying
    print(f"Missing {missing} - downloading only those" if DATASET else
          "No weights dataset attached - downloading everything (attach one to skip this next time).")
    subprocess.run([sys.executable, "/tmp/hy3d/prefetch.py", WEIGHTS], check=True)
OFFLINE = True  # every file is present locally now
print("Weights:", WEIGHTS)
'''),
    md("## 4. Worker scripts\nEach one runs in its own process, pinned to one GPU with `CUDA_VISIBLE_DEVICES`."),
    writefile("/tmp/hy3d/shape_worker.py", SHAPE_WORKER_PY),
    writefile("/tmp/hy3d/paint_worker.py", PAINT_WORKER_PY),
    md("## 5. Settings and input images"),
    code(r'''
import glob, os, re, shutil

REPO = "/tmp/Hunyuan3D-2.1"
OUT_DIR = "/kaggle/working/hy3d_out"

SETTINGS = dict(
    # shape (GPU 0)
    steps=30,                # 30 is the official demo default; 50 is slightly sharper and slower
    guidance_scale=5.0,
    octree_resolution=256,   # 384/512 = finer geometry, more VRAM and time
    num_chunks=8000,
    seed=1234,
    max_faces=40000,         # faces in the final model; 200000 keeps fingers and faces much sharper (slower texturing)
    force_rembg=False,       # True = always strip the background, even for RGBA inputs
    # texture (GPU 1). The official demo uses 8 views @ 768 on a 24 GB card; a 16 GB T4 is safe at 6 @ 512.
    texture_views=6,         # 6-9
    texture_resolution=512,  # 512 or 768
    skip_existing=True,      # re-running only does images that aren't finished yet
)
MAX_IMAGES = 50

# Every image in attached Kaggle inputs; falls back to the repo's demo image.
exts = (".png", ".jpg", ".jpeg", ".webp")
paths = sorted(p for p in glob.glob("/kaggle/input/**/*", recursive=True)
               if p.lower().endswith(exts) and not (DATASET and p.startswith(DATASET + "/")))
if not paths:
    paths = [f"{REPO}/assets/demo.png"]
    print("No images found in /kaggle/input - using the demo image.")

images, seen = [], set()
for p in paths[:MAX_IMAGES]:
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", os.path.splitext(os.path.basename(p))[0])
    base, i = name, 1
    while name in seen:  # two inputs with the same file name get separate output folders
        i += 1
        name = f"{base}_{i}"
    seen.add(name)
    images.append({"name": name, "path": p})

print(f"{len(images)} image(s):", ", ".join(x["name"] for x in images))
'''),
    md("## 6. Run on both GPUs\nThe shape worker starts first. The paint worker starts once the shape model is loaded, so both models aren't loading at the same time against Kaggle's ~30 GB of RAM. A GPU status line prints every 30 s, so you can see both cards busy."),
    code(r'''
import json, os, shutil, subprocess, sys, time

WORK = "/tmp/hy3d"
QUEUE = f"{WORK}/queue"
# A rerun starts clean: no leftover workers from an interrupted run, no stale logs printed as if new.
subprocess.run(["pkill", "-f", f"{WORK}/(shape|paint)_worker.py"])
for script in ("shape_worker", "paint_worker"):
    if os.path.exists(f"{WORK}/{script}.log"):
        os.remove(f"{WORK}/{script}.log")
shutil.rmtree(QUEUE, ignore_errors=True)
os.makedirs(QUEUE)
os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{WORK}/config.json", "w") as f:
    json.dump(dict(SETTINGS, repo=REPO, queue=QUEUE, out=OUT_DIR, weights=WEIGHTS, images=images), f)

n_gpus = len(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip().splitlines())
paint_gpu = 1 if n_gpus >= 2 else 0
print(f"{n_gpus} GPU(s): shape -> cuda:0, paint -> cuda:{paint_gpu}"
      + ("" if n_gpus >= 2 else " (single GPU: stages run one after another)"))


def launch(script, gpu):
    log = open(f"{WORK}/{script}.log", "w")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
               HY3DGEN_MODELS=WEIGHTS,             # where hy3dshape looks for tencent/Hunyuan3D-2.1
               U2NET_HOME=f"{WEIGHTS}/u2net")      # where rembg looks for u2net.onnx
    if OFFLINE:
        env["HF_HUB_OFFLINE"] = "1"  # everything is local: fail loudly rather than download
    return subprocess.Popen([sys.executable, f"{WORK}/{script}.py", f"{WORK}/config.json"],
                            stdout=log, stderr=subprocess.STDOUT, env=env)


def read_gb(proc):
    """Bytes a worker has read so far - shows model loading progress while the GPUs sit at 0%."""
    try:
        return int(open(f"/proc/{proc.pid}/io").read().split("rchar:")[1].split()[0]) / 2**30
    except (OSError, IndexError, ValueError):
        return 0.0


offsets = {}
def pump(script):
    """Print new log lines, skipping progress-bar noise."""
    path = f"{WORK}/{script}.log"
    if not os.path.exists(path):
        return
    with open(path, errors="replace") as f:
        f.seek(offsets.get(script, 0))
        chunk = f.read()
        offsets[script] = f.tell()
    for line in chunk.replace("\r", "\n").splitlines():
        if line.strip() and not any(s in line for s in ("it/s]", "s/it]", "%|")):
            print(line)


start = time.time()
shape = launch("shape_worker", 0)
paint = None
last_stats = 0
while True:
    pump("shape_worker")
    pump("paint_worker")
    if shape.poll() is not None:
        open(f"{QUEUE}/_SHAPE_DONE", "a").close()  # also unblocks paint if shape crashed
        if shape.returncode != 0 and not os.path.exists(f"{QUEUE}/_SHAPE_READY"):
            print("Shape worker crashed while loading - see the log above.")
            break
    if paint is None and ((n_gpus >= 2 and os.path.exists(f"{QUEUE}/_SHAPE_READY")) or shape.poll() is not None):
        paint = launch("paint_worker", paint_gpu)
    if paint is not None and paint.poll() is not None and shape.poll() is not None:
        break
    if time.time() - last_stats > 30:
        last_stats = time.time()
        stats = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        gpus = "  ".join(f"cuda:{i} {u.strip()}% {int(m) / 1024:.1f}GB"
                         for i, u, m in (l.split(",") for l in stats.strip().splitlines()))
        reads = "  ".join(f"{n} read {read_gb(p):.1f}GB" for n, p in (("shape", shape), ("paint", paint))
                          if p is not None and p.poll() is None)
        print(f"--- {time.time() - start:5.0f}s | {gpus} | {reads}")
    time.sleep(3)

pump("shape_worker")
pump("paint_worker")
print(f"Finished in {(time.time() - start) / 60:.1f} min (shape exit {shape.returncode}, "
      f"paint exit {None if paint is None else paint.returncode})")
'''),
    md("## 7. Results"),
    code(r'''
import glob, os, shutil
from IPython.display import FileLink, display

done = sorted(glob.glob(f"{OUT_DIR}/*/textured_mesh.glb"))
shape_only = sorted(set(glob.glob(f"{OUT_DIR}/*/white_mesh.glb")) - {p.replace("textured_mesh", "white_mesh") for p in done})
print(f"{len(done)} textured model(s), {len(shape_only)} untextured only")
for p in done:
    print(f"  {os.path.getsize(p) / 2**20:6.1f} MB  {p}")

shutil.make_archive("/kaggle/working/hy3d_results", "zip", OUT_DIR)
display(FileLink("hy3d_results.zip"))
'''),
    md("### Optional: preview a model in the notebook"),
    code(r'''
import base64
from IPython.display import HTML

if done:
    glb = done[0]  # change the index to preview another model
    data = base64.b64encode(open(glb, "rb").read()).decode()
    display(HTML(f"""
    <script type="module" src="https://unpkg.com/@google/model-viewer@3.5.0/dist/model-viewer.min.js"></script>
    <p>{glb}</p>
    <model-viewer src="data:model/gltf-binary;base64,{data}" camera-controls auto-rotate
                  style="width:100%;height:520px;background:#222"></model-viewer>"""))
'''),
    md("""
## Troubleshooting
- **`OUT OF GPU MEMORY` in the paint log**: keep `texture_resolution=512` and lower `texture_views` to 6. Shape-stage OOM: lower `octree_resolution` or `num_chunks`.
- **`nvcc not found`**: the session has no GPU attached. Set the accelerator to GPU T4 x2 and restart.
- **Install fails on a package**: Kaggle updates its image regularly. The error usually names the package; loosen that pin in `install.sh` and rerun the cell.
- **Session restarted**: `/tmp` is wiped, so rerun everything. Outputs in `/kaggle/working` survive while the session is alive, and `skip_existing=True` resumes where it stopped.
- Full worker logs: `/tmp/hy3d/shape_worker.log` and `/tmp/hy3d/paint_worker.log`.
"""),
]

builder_cells = [
    md("""
# Hunyuan3D 2.1 weights → Kaggle dataset (run once)

Downloads every file the main notebook needs (**17.8 GB**) into `/kaggle/working/hy3d-weights`. You then turn this
notebook's output into a private dataset. After that, attaching the dataset to `hunyuan3d_2_1_kaggle_dual_t4.ipynb`
means nothing gets downloaded again.

**This needs no GPU,** so building it doesn't use any of your weekly GPU hours.

1. **Settings → Accelerator → None**, **Internet → On**
2. **Save Version → Save & Run All (Commit)**, then wait for the version to finish (about 10–20 minutes).
3. Create the dataset from the output. Either:
   - open the finished version, go to the **Output** tab, and choose **New Dataset**, or
   - go to **Datasets → New Dataset**, pick **Notebook Output Files**, and select this notebook.

   Name it, for example, `hunyuan3d-2-1-weights`, and keep it **Private**.
   Already have an incomplete dataset? Open it, choose **New Version**, and pick this notebook's output instead.
4. In the main notebook: **Add Input → Your Datasets → hunyuan3d-2-1-weights.** It's found automatically.

Kaggle's output limit is 20 GB, so don't write anything else to `/kaggle/working` in this notebook.
"""),
    code("!pip install -q --root-user-action=ignore hf_transfer"),
    writefile("/tmp/prefetch.py", PREFETCH_PY),
    code("""
import subprocess, sys
# check=True fails the commit if anything is missing, so an incomplete output never becomes a dataset.
subprocess.run([sys.executable, "/tmp/prefetch.py", "/kaggle/working/hy3d-weights"], check=True)
!du -sh /kaggle/working/hy3d-weights/* /kaggle/working/hy3d-weights/*/*
"""),
]


def notebook(cells, kaggle_meta):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": kaggle_meta,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"cell-{i:02d}"
        c["source"] = c["source"].splitlines(keepends=True)
    return nb


out_dir = sys.argv[1]
for fname, nb in [
    ("hunyuan3d_2_1_kaggle_dual_t4.ipynb",
     notebook(cells, {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True})),
    ("hunyuan3d_2_1_weights_builder.ipynb",
     notebook(builder_cells, {"accelerator": "none", "isInternetEnabled": True, "isGpuEnabled": False})),
]:
    with open(f"{out_dir}/{fname}", "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("wrote", fname)

# Also emit the helper scripts so they can be tested outside the notebook.
if len(sys.argv) > 2:
    import os
    os.makedirs(sys.argv[2], exist_ok=True)
    for fname, body in [("install.sh", INSTALL_SH), ("prefetch.py", PREFETCH_PY),
                        ("shape_worker.py", SHAPE_WORKER_PY), ("paint_worker.py", PAINT_WORKER_PY)]:
        with open(os.path.join(sys.argv[2], fname), "w", encoding="utf-8", newline="\n") as f:
            f.write(body.lstrip("\n"))
