"""Measure container resources, real mounts, and a CUDA/checkpoint smoke run."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


def command(argv):
    """Capture a bounded, read-only system command without environment secrets."""
    p = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}


def read_limit(name):
    p = Path('/sys/fs/cgroup') / name
    return p.read_text().strip() if p.exists() else None


def mount_info(path):
    """Resolve symlinks before checking mounts; never write to unmounted paths."""
    p = Path(path).resolve()
    result = {"path": path, "resolved": str(p), "exists": p.exists()}
    if not p.exists():
        result['verified_mount'] = False
        return result
    result['findmnt'] = command(['findmnt', '-J', '-T', str(p)])
    fs = json.loads(result['findmnt']['stdout'])['filesystems'][0]
    result['filesystem'] = fs
    result['verified_mount'] = Path(fs['target']).resolve() == p and fs['fstype'] not in ('overlay', 'tmpfs')
    result['space_bytes'] = dict(zip(('total', 'used', 'free'), shutil.disk_usage(p)))
    st = os.statvfs(p)
    result['inodes'] = {'total': st.f_files, 'free': st.f_favail}
    if result['verified_mount'] and path != '/':
        with tempfile.NamedTemporaryFile(dir=p, prefix='.tsp-probe-', delete=True) as f:
            f.write(b'tsp-storage-probe\n')
            f.flush()
            os.fsync(f.fileno())
            result['write_read_verified'] = Path(f.name).read_bytes() == b'tsp-storage-probe\n'
    return result


def cuda_smoke(work):
    """Exercise attention training, optimizer/checkpoint restore, and tiny Qwen."""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    assert torch.cuda.is_available(), 'CUDA unavailable'
    torch.manual_seed(20261116)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    model = torch.nn.TransformerEncoderLayer(128, 8, 256, batch_first=True, dropout=0).cuda()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(4, 100, 128, device='cuda')
    loss = model(x).square().mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optim.step()
    with torch.no_grad():
        expected = model(x).clone()
    with tempfile.NamedTemporaryFile(dir=work, suffix='.pt') as f:
        torch.save({'model': model.state_dict(), 'optimizer': optim.state_dict()}, f.name)
        checkpoint = torch.load(f.name, map_location='cuda', weights_only=True)
        restored = torch.nn.TransformerEncoderLayer(128, 8, 256, batch_first=True, dropout=0).cuda()
        restored.load_state_dict(checkpoint['model'])
        optim2 = torch.optim.AdamW(restored.parameters(), lr=1e-3)
        optim2.load_state_dict(checkpoint['optimizer'])
        with torch.no_grad():
            err = (restored(x) - expected).abs().max().item()
    assert err < 1e-6, err
    config = Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=128, eos_token_id=2, bos_token_id=1, pad_token_id=0)
    qwen = Qwen2ForCausalLM(config).cuda()
    ids = torch.randint(3, 256, (1, 16), device='cuda')
    qloss = qwen(ids, labels=ids).loss
    qloss.backward()
    assert torch.isfinite(qloss)
    out = qwen.generate(ids, max_new_tokens=2, do_sample=False)
    torch.cuda.synchronize()
    return {'success': True, 'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
            'device': torch.cuda.get_device_name(0), 'device_count': torch.cuda.device_count(),
            'total_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
            'attention_loss': loss.item(), 'checkpoint_max_abs_error': err,
            'tiny_random_qwen_loss': qloss.item(), 'tiny_random_qwen_output_shape': list(out.shape),
            'elapsed_seconds': time.perf_counter() - start,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'scope': 'Small random networks only; does not validate full 7B loading or GRPO'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = {'measured_at_utc': datetime.now(timezone.utc).isoformat(),
            'hostname': platform.node(), 'python': platform.python_version(),
            'python_executable': os.sys.executable, 'platform': platform.platform(),
            'host_logical_cpu_count': os.cpu_count(),
            'cpu_max': read_limit('cpu.max'), 'memory_max': read_limit('memory.max'),
            'memory_current': read_limit('memory.current'),
            'cpuset_cpus_effective': read_limit('cpuset.cpus.effective'),
            'nvidia_smi': command(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv,noheader']),
            'lscpu': command(['lscpu', '-J']),
            'mounts': [mount_info(p) for p in ('/', '/root/autodl-tmp', '/root/autodl-fs')],
            'tmux': command(['tmux', '-V']),
            'cache_directories': {k: os.getenv(k) for k in ('HF_HOME','HF_HUB_CACHE','PIP_CACHE_DIR','TORCH_HOME','TMPDIR','XDG_CACHE_HOME')},
            'packages': {p: importlib.metadata.version(p) for p in
                         ('torch','numpy','scipy','pandas','matplotlib','transformers','tokenizers','accelerate','huggingface-hub','safetensors','gdown','pytest')}}
    try:
        data['cuda_smoke'] = cuda_smoke(args.output.parent)
    except Exception as e:
        data['cuda_smoke'] = {'success': False, 'error': repr(e)}
    args.output.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'cuda_smoke': data['cuda_smoke'],
                      'cpu_max': data['cpu_max'], 'memory_max': data['memory_max'],
                      'mounts': [{k: m.get(k) for k in ('path','verified_mount','write_read_verified','space_bytes')} for m in data['mounts']]}, indent=2))
    if not data['cuda_smoke']['success']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
