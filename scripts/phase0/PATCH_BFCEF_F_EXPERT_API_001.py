from pathlib import Path
import shutil
import time
import py_compile
import hashlib

ROOT = Path(r"D:/MARIVISION/H0_workbench/eval_tools/bfcef_f/experts")
BACKUP = (
    ROOT.parent.parent
    / ("_expert_api_backup_" + time.strftime("%Y%m%d_%H%M%S"))
)

FILES = {}

FILES["scale.py"] = '''import torch, torch.nn as nn
from ..evidence_builders import ScaleEvidenceBuilder

class ScaleSpecialist(nn.Module):
    def __init__(self, d_model=256, d_hidden=64, d_ctx=32):
        super().__init__()
        self.evidence_builder = ScaleEvidenceBuilder(d_model, d_hidden)
        self.proj_cls = nn.Linear(d_hidden, d_model)
        self.proj_box = nn.Linear(d_hidden, d_model)
        self.adapter_cls = nn.Linear(d_ctx, d_hidden)
        self.adapter_box = nn.Linear(d_ctx, d_hidden)
        self._zero()

    def _zero(self):
        for m in (
            self.proj_cls,
            self.proj_box,
            self.adapter_cls,
            self.adapter_box,
        ):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, h, cues, b, spatial, C):
        f = self.evidence_builder(h, cues, b, spatial)
        fc = f * (1 + torch.tanh(self.adapter_cls(C)))
        fb = f * (1 + torch.tanh(self.adapter_box(C)))
        return self.proj_cls(fc), self.proj_box(fb), f
'''

FILES["localization.py"] = '''import torch, torch.nn as nn
from ..evidence_builders import LocalizationEvidenceBuilder

class LocalizationSpecialist(nn.Module):
    def __init__(self, d_model=256, d_hidden=64, d_ctx=32):
        super().__init__()
        self.evidence_builder = LocalizationEvidenceBuilder(d_model, d_hidden)
        self.proj_cls = nn.Linear(d_hidden, d_model)
        self.proj_box = nn.Linear(d_hidden, d_model)
        self.expected_iou_head = nn.Sequential(
            nn.Linear(d_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.adapter_cls = nn.Linear(d_ctx, d_hidden)
        self.adapter_box = nn.Linear(d_ctx, d_hidden)
        self._zero()

    def _zero(self):
        for m in (
            self.proj_cls,
            self.proj_box,
            self.adapter_cls,
            self.adapter_box,
        ):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, h, cues, b, r, spatial, C):
        f = self.evidence_builder(h, cues, b, r, spatial)
        fc = f * (1 + torch.tanh(self.adapter_cls(C)))
        fb = f * (1 + torch.tanh(self.adapter_box(C)))
        return (
            self.proj_cls(fc),
            self.proj_box(fb),
            f,
            self.expected_iou_head(f),
        )
'''

FILES["semantic.py"] = '''import torch, torch.nn as nn
from ..evidence_builders import SemanticEvidenceBuilder

class SemanticSpecialist(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_hidden=64,
        d_ctx=32,
        num_classes=7,
    ):
        super().__init__()
        self.evidence_builder = SemanticEvidenceBuilder(
            d_model,
            num_classes,
            d_hidden,
        )
        self.proj_cls = nn.Linear(d_hidden, d_model)
        self.proj_box = nn.Linear(d_hidden, d_model)
        self.adapter_cls = nn.Linear(d_ctx, d_hidden)
        self.adapter_box = nn.Linear(d_ctx, d_hidden)
        self._zero()

    def _zero(self):
        for m in (
            self.proj_cls,
            self.proj_box,
            self.adapter_cls,
            self.adapter_box,
        ):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, h, cues, z, C):
        f = self.evidence_builder(h, cues, z)
        fc = f * (1 + torch.tanh(self.adapter_cls(C)))
        fb = f * (1 + torch.tanh(self.adapter_box(C)))
        return self.proj_cls(fc), self.proj_box(fb), f
'''

FILES["temporal.py"] = '''import torch, torch.nn as nn
from ..evidence_builders import TemporalEvidenceBuilder
from ..contracts import TemporalState

class TemporalSpecialist(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_hidden=64,
        d_ctx=32,
        K=3,
    ):
        super().__init__()
        self.evidence_builder = TemporalEvidenceBuilder(
            d_model,
            K,
            d_hidden,
        )
        self.proj_cls = nn.Linear(d_hidden, d_model)
        self.proj_box = nn.Linear(d_hidden, d_model)
        self.adapter_cls = nn.Linear(d_ctx, d_hidden)
        self.adapter_box = nn.Linear(d_ctx, d_hidden)
        self._zero()

    def _zero(self):
        for m in (
            self.proj_cls,
            self.proj_box,
            self.adapter_cls,
            self.adapter_box,
        ):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, h, t: TemporalState, C):
        f = self.evidence_builder(h, t)
        fc = f * (1 + torch.tanh(self.adapter_cls(C)))
        fb = f * (1 + torch.tanh(self.adapter_box(C)))
        return self.proj_cls(fc), self.proj_box(fb), f
'''

if not ROOT.exists():
    raise SystemExit(f"[FATAL] Missing expert directory: {ROOT}")

BACKUP.mkdir(parents=True)

for rel, text in FILES.items():
    dst = ROOT / rel

    if dst.exists():
        shutil.copy2(dst, BACKUP / rel)

    dst.write_text(
        text,
        encoding="utf-8",
        newline="\n",
    )

    py_compile.compile(
        str(dst),
        doraise=True,
    )

    sha = hashlib.sha256(
        dst.read_bytes()
    ).hexdigest()

    print(f"[PASS] {rel}")
    print(f"       SHA256: {sha}")

print()
print("[PASS] Expert-wrapper API synchronized.")
print(f"[INFO] Backup saved to: {BACKUP}")