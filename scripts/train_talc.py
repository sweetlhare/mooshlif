"""Обучение U-Net детекции талька на слабой разметке (синие обводки).

Стратегия работы со слабой разметкой (см. ответы организаторов: контуры
не замыкаются, внутрь областей попадают сульфиды):
- позитив = залитая область обводки МИНУС яркие фазы (сульфид/серая) —
  тальк по определению тёмная фаза;
- негатив с весом 1.0 = яркие фазы везде + вся площадь снимков из папок
  рядовых/труднообогатимых руд (по метке папки талька там нет);
- негатив с весом 0.4 = матрица вне обводок на оталькованных снимках
  (разметка может быть неисчерпывающей);
- жёсткая цветовая аугментация (перенос оливковый→тёмный домен).

Запуск:
    python scripts/train_talc.py --masks <dir> --epochs 25 --out models/talc_unet.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shlifscan.config import PipelineConfig
from shlifscan.imio import imread_rgb
from shlifscan.preprocess import preprocess
from shlifscan.segment import segment_phases

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ANNOT_DIR = DATA / "Фото руд по сортам. ч1/Оталькованные руды"
NEG_DIRS = [
    DATA / "Фото руд по сортам. ч1/Рядовые руды",
    DATA / "Фото руд по сортам. ч1/Труднообогатимые руды",
    DATA / "Фото руд по сортам. ч2/рядовые",
    DATA / "Фото руд по сортам. ч2/тонкие",
]

WORK_SIDE = 1536  # рабочая длинная сторона при подготовке примеров


def sample_group(name: str) -> str:
    """Группа образца по имени файла (для group-split без утечек)."""
    stem = Path(name).stem
    for sep in (" ", "-", "_"):
        if sep in stem:
            return stem.split(sep)[0]
    return stem


def load_failed_names(masks_dir: Path) -> set[str]:
    """Имена снимков с ненадёжными масками из report.csv рядом с масками."""
    for report in (masks_dir / "report.csv", masks_dir.parent / "report.csv"):
        if report.exists():
            import csv

            with open(report, encoding="utf-8") as f:
                return {
                    Path(row["name"]).stem
                    for row in csv.DictReader(f)
                    if "failed" in row.get("status", "")
                }
    return set()


def prepare_examples(masks_dir: Path, out_dir: Path, n_neg: int = 40,
                     seed: int = 0, dense: bool = False) -> list[dict]:
    """Готовит npz-примеры: image (uint8 HxWx3), target (0/1), weight (f32)."""
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = PipelineConfig()
    index = []
    failed = load_failed_names(masks_dir)
    if failed:
        print(f"исключены ненадёжные маски: {sorted(failed)}")

    def work_resize(arr, interp):
        h, w = arr.shape[:2]
        s = WORK_SIDE / max(h, w)
        return cv2.resize(arr, (int(w * s), int(h * s)), interpolation=interp)

    def disk(r):
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

    # --- референсные LAB-статистики (до обработки: нужны для нормализации) ---
    from shlifscan.preprocess import lab_stats, reinhard_to_reference

    mask_files_all = [f for f in sorted(masks_dir.glob("*.png")) if f.stem not in failed]
    stats = []
    for mf in mask_files_all[:20]:
        img_path = ANNOT_DIR / (mf.stem + ".JPG")
        if img_path.exists():
            stats.append(lab_stats(work_resize(imread_rgb(img_path), cv2.INTER_AREA)))
    ref = {
        "mean": [float(np.mean([s["mean"][i] for s in stats])) for i in range(3)],
        "std": [float(np.mean([s["std"][i] for s in stats])) for i in range(3)],
    }
    (ROOT / "models" / "talc_ref_stats.json").write_text(
        json.dumps(ref, indent=1), encoding="utf-8")
    print("референс LAB:", ref)

    # --- позитивные (оталькованные с масками): seed-карта слабой разметки ---
    # target: 1 = уверенный тальк, 0 = уверенный не-тальк, weight = 0 → ignore.
    # Принципы (ScribbleBench/PU): ядро области после эрозии — позитив;
    # кольцо у границы и разрывы — ignore; яркие фазы — негатив с ignore-каймой;
    # матрица ВНЕ обводок на оталькованном снимке — ignore (разметка
    # неисчерпывающая, снаружи тоже может быть тальк).
    mask_files = mask_files_all
    for mf in mask_files:
        img_path = ANNOT_DIR / (mf.stem + ".JPG")
        if not img_path.exists():
            print(f"! нет оригинала для {mf.name}")
            continue
        rgb = work_resize(imread_rgb(img_path), cv2.INTER_AREA)
        region_raw = cv2.imread(str(mf), 0)
        if region_raw.shape[:2] != rgb.shape[:2]:
            region_raw = cv2.resize(region_raw, (rgb.shape[1], rgb.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
        region = region_raw > 127

        pre = preprocess(rgb, cfg.preprocess, downscale=False)
        rgb = reinhard_to_reference(rgb, ref)  # хранится нормализованный вход
        seg = segment_phases(pre, cfg.segment)
        bright = seg.sulfide | seg.gray

        # dense-маски (SAMRefiner): граница уже по фазам — эрозия минимальна
        erode_r = 1 if dense else 6
        region_core = cv2.erode(region.astype(np.uint8), disk(erode_r)).astype(bool)
        fg = region_core & ~cv2.dilate(bright.astype(np.uint8), disk(2)).astype(bool)

        target = np.zeros(region.shape, np.uint8)
        target[fg] = 1
        weight = np.zeros(region.shape, np.float32)        # по умолчанию ignore
        weight[fg] = 1.0                                   # тальк — позитив
        weight[bright] = 1.0                               # яркие фазы — негатив
        # ignore-кайма вокруг ярких фаз (смешанные пиксели)
        halo = cv2.dilate(bright.astype(np.uint8), disk(2)).astype(bool) & ~bright
        weight[halo] = 0.0
        weight[~pre.valid] = 0.0

        # эталон для метрики ДОЛИ (без эрозии): область минус яркие фазы
        eval_target = (region & ~bright).astype(np.uint8)
        valid = pre.valid.astype(np.uint8)

        name = f"pos_{mf.stem}.npz"
        np.savez_compressed(out_dir / name, image=rgb, target=target, weight=weight,
                            eval_target=eval_target, valid=valid)
        index.append({
            "name": name, "kind": "pos", "group": sample_group(mf.name),
            "talc_frac_region": float(region.mean()),
            "talc_frac_clean": float(eval_target.mean()),
        })

    # --- негативные (рядовые/тонкие, талька нет по метке папки) ---
    # дедупликация: в ч2 встречаются побайтовые копии аннотированных
    # оталькованных снимков ч1 — исключаем по md5; DSCN-имена в ч2 — копии
    # оливкового домена с сомнительными метками, тоже исключаем
    import hashlib

    pos_md5 = {
        hashlib.md5((ANNOT_DIR / (mf.stem + ".JPG")).read_bytes()).hexdigest()
        for mf in mask_files if (ANNOT_DIR / (mf.stem + ".JPG")).exists()
    }
    neg_files = []
    for d in NEG_DIRS:
        files = [f for f in sorted(d.iterdir())
                 if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
                 and not ("ч2" in str(d) and f.stem.upper().startswith("DSCN"))]
        neg_files += rng.sample(files, min(n_neg // len(NEG_DIRS) + 2, len(files)))
    neg_files = [f for f in neg_files
                 if hashlib.md5(f.read_bytes()).hexdigest() not in pos_md5]
    rng.shuffle(neg_files)
    neg_files = neg_files[:n_neg]

    for f in neg_files:
        rgb = work_resize(imread_rgb(f), cv2.INTER_AREA)
        pre = preprocess(rgb, cfg.preprocess, downscale=False)
        rgb = reinhard_to_reference(rgb, ref)
        target = np.zeros(pre.ln.shape, np.uint8)
        weight = np.full(target.shape, 0.7, np.float32)
        weight[~pre.valid] = 0.0
        name = f"neg_{f.parent.name}_{f.stem}.npz".replace(" ", "_")
        np.savez_compressed(out_dir / name, image=rgb, target=target, weight=weight,
                            eval_target=target, valid=pre.valid.astype(np.uint8))
        index.append({"name": name, "kind": "neg",
                      "group": "neg_" + sample_group(f.name)})

    (out_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"подготовлено: {sum(1 for i in index if i['kind']=='pos')} pos, "
          f"{sum(1 for i in index if i['kind']=='neg')} neg")
    return index


# ---------------------------------------------------------------- обучение
def build_augmentation(crop: int):
    import albumentations as A

    return A.Compose([
        # масштабная аугментация: магнификации 5x/10x/20x (чувствительность
        # к масштабу подтверждена диагностикой v1)
        A.RandomScale(scale_limit=(-0.5, 0.5), p=0.7),
        A.PadIfNeeded(crop, crop, border_mode=4),
        A.RandomCrop(crop, crop),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # цветовой сдвиг вокруг Reinhard-якоря (умеренный: домены уже сведены)
        A.RandomBrightnessContrast(brightness_limit=(-0.35, 0.25),
                                   contrast_limit=(-0.3, 0.3), p=0.8),
        A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=30,
                             val_shift_limit=25, p=0.8),
        A.RandomGamma(gamma_limit=(70, 145), p=0.5),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ])


class TalcDataset:
    def __init__(self, files: list[Path], crop: int = 512, samples_per_image: int = 8):
        self.files = files
        self.crop = crop
        self.spi = samples_per_image
        self.aug = build_augmentation(crop)
        self._cache: dict[int, dict] = {}

    def __len__(self):
        return len(self.files) * self.spi

    def _load(self, fi: int) -> dict:
        if fi not in self._cache:
            z = np.load(self.files[fi])
            self._cache[fi] = {k: z[k] for k in ("image", "target", "weight")}
        return self._cache[fi]

    def __getitem__(self, i: int):
        import torch

        d = self._load(i % len(self.files))
        a = self.aug(image=d["image"], masks=[d["target"], d["weight"]])
        img = a["image"].astype(np.float32) / 255.0
        tgt, wgt = a["masks"][0].astype(np.float32), a["masks"][1].astype(np.float32)
        return (
            torch.from_numpy(img.transpose(2, 0, 1)),
            torch.from_numpy(tgt)[None],
            torch.from_numpy(wgt)[None],
        )


def gated_crf_loss(logits, img, radius: int = 5, sigma_xy: float = 3.0,
                   sigma_rgb: float = 0.1, down: int = 4):
    """Компактный Gated CRF loss (Obukhov 2019): штраф за разность предсказаний
    у близких по цвету и координатам пикселей. Считается на даунскейле."""
    import torch
    import torch.nn.functional as F

    p = torch.sigmoid(logits)
    p = F.avg_pool2d(p, down)
    g = F.avg_pool2d(img, down)
    b, _, h, w = p.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=p.device, dtype=p.dtype),
        torch.arange(w, device=p.device, dtype=p.dtype), indexing="ij")
    coords = torch.stack([yy, xx])[None].expand(b, -1, -1, -1)
    feat = torch.cat([g, coords / sigma_xy], 1)  # цвет + координаты

    k = 2 * radius + 1
    feat_u = F.unfold(feat, k, padding=radius).view(b, feat.shape[1], k * k, h * w)
    p_u = F.unfold(p, k, padding=radius).view(b, 1, k * k, h * w)
    center = feat.view(b, feat.shape[1], 1, h * w)
    d2 = ((feat_u - center) ** 2)
    w_rgb = torch.exp(-d2[:, :3].sum(1) / (2 * sigma_rgb ** 2))
    w_xy = torch.exp(-d2[:, 3:].sum(1) / 2.0)
    kernel = (w_rgb * w_xy)  # b, k*k, h*w
    pc = p.view(b, 1, 1, h * w)
    psi = pc * (1 - p_u) + (1 - pc) * p_u  # несогласие предсказаний
    return (kernel[:, None] * psi).mean()


def train(args) -> None:
    import torch
    import segmentation_models_pytorch as smp
    from torch.utils.data import DataLoader

    device = "mps" if torch.backends.mps.is_available() else \
        "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    prep_dir = Path(args.prep_dir)
    index = json.loads((prep_dir / "index.json").read_text(encoding="utf-8"))

    # group-split: холдаут по группам образцов среди позитивных
    pos_groups = sorted({i["group"] for i in index if i["kind"] == "pos"})
    rng = random.Random(42)
    val_groups = set(rng.sample(pos_groups, max(2, int(len(pos_groups) * args.val_frac))))
    train_files, val_files = [], []
    for item in index:
        f = prep_dir / item["name"]
        if item["kind"] == "pos" and item["group"] in val_groups:
            val_files.append(f)
        else:
            train_files.append(f)
    print(f"train {len(train_files)} файлов, val {len(val_files)} (группы: {sorted(val_groups)})")

    ds_tr = TalcDataset(train_files, crop=args.crop, samples_per_image=args.spi)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=0)

    model = smp.Unet(args.encoder, encoder_weights="imagenet", classes=1)
    if args.init_encoder:
        ckpt = torch.load(args.init_encoder, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        enc_sd = {k[len("encoder."):]: v for k, v in sd.items()
                  if k.startswith("encoder.")}
        model.encoder.load_state_dict(enc_sd, strict=False)
        print(f"init encoder из {args.init_encoder}: {len(enc_sd)} тензоров")
    model = model.to(device)
    if args.freeze_encoder_epochs > 0:
        for prm in model.encoder.parameters():
            prm.requires_grad = False
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(dl_tr))
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    q = 0.7
    a_agce = float(getattr(args, "agce_a", 4.0))
    loss_kind = getattr(args, "loss", "gce")

    def _fg_term(p_fg):
        """Робастный к шуму метки FG-терм по выбранному лоссу.
        gce:  (1 - p^q)/q                     (Zhang&Sabuncu 2018)
        agce: ((a+1)^q - (a+p)^q)/q           (Zhou 2021, асимметричный — под наш
              шум «сульфид внутри контура помечен тальком»)
        tloss: робастный Student-t NLL остатка (González-Jiménez 2023)"""
        if loss_kind == "agce":
            return (((a_agce + 1.0) ** q - (a_agce + p_fg) ** q) / q).mean()
        if loss_kind == "tloss":
            # residual = 1 - p (тянем p→1); Student-t NLL с ν=2 (тяжёлые хвосты
            # авто-даунвесят выбросы-мислейблы)
            nu = 2.0
            r2 = (1.0 - p_fg) ** 2
            return ((nu + 1.0) / 2.0 * torch.log1p(r2 / nu)).mean()
        return ((1 - p_fg ** q) / q).mean()          # gce (по умолчанию)

    def weak_loss(logits, target, weight, img):
        """pCE(ignore по weight=0) + робастный FG-терм + Gated CRF."""
        p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
        fg = (target > 0.5) & (weight > 0)
        bg = (target <= 0.5) & (weight > 0)
        loss_fg = _fg_term(p[fg]) if fg.any() else logits.sum() * 0
        loss_bg = (bce(logits, target) * weight)[bg].mean() if bg.any() else logits.sum() * 0
        return loss_fg + loss_bg + 0.15 * gated_crf_loss(logits, img)

    best_mae = 1e9
    history = []
    for ep in range(args.epochs):
        if args.freeze_encoder_epochs and ep == args.freeze_encoder_epochs:
            for prm in model.encoder.parameters():
                prm.requires_grad = True
            print(f"ep {ep}: энкодер разморожен")
        model.train()
        tot = 0.0
        for img, tgt, wgt in dl_tr:
            img, tgt, wgt = img.to(device), tgt.to(device), wgt.to(device)
            logits = model(img)
            loss = weak_loss(logits, tgt, wgt, img)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss)
        mae, iou = evaluate(model, val_files, device, args)
        history.append({"epoch": ep, "loss": tot / len(dl_tr),
                        "val_frac_mae": mae, "val_iou": iou})
        print(f"ep {ep:02d} loss {tot/len(dl_tr):.4f} | val MAE доли {mae*100:.2f}% "
              f"| val IoU {iou:.3f}", flush=True)
        if mae < best_mae:
            best_mae = mae
            save_checkpoint(model, args, history)
    print(f"best val MAE доли талька: {best_mae*100:.2f}% → {args.out}")


@np.errstate(invalid="ignore")
def evaluate(model, val_files, device, args) -> tuple[float, float]:
    """MAE доли талька и IoU на held-out снимках (полный кадр тайлами)."""
    import torch

    model.eval()
    maes, ious = [], []
    with torch.no_grad():
        for f in val_files:
            z = np.load(f)
            img = z["image"].astype(np.float32) / 255.0
            # метрика доли — против неэродированного эталона на валидной области
            tgt = (z["eval_target"] if "eval_target" in z else z["target"]) > 0
            wgt = (z["valid"] if "valid" in z else z["weight"]) > 0
            h, w = img.shape[:2]
            prob = np.zeros((h, w), np.float32)
            cnt = np.zeros((h, w), np.float32)
            t = args.crop
            for y in range(0, h, t - 64):
                for x in range(0, w, t - 64):
                    y1, x1 = min(y + t, h), min(x + t, w)
                    y0, x0 = max(y1 - t, 0), max(x1 - t, 0)
                    patch = img[y0:y1, x0:x1]
                    ph, pw = patch.shape[:2]
                    if ph < t or pw < t:
                        patch = np.pad(patch, ((0, t - ph), (0, t - pw), (0, 0)), mode="reflect")
                    inp = torch.from_numpy(patch.transpose(2, 0, 1))[None].to(device)
                    out = torch.sigmoid(model(inp))[0, 0].cpu().numpy()[:ph, :pw]
                    prob[y0:y0+ph, x0:x0+pw] += out
                    cnt[y0:y0+ph, x0:x0+pw] += 1
            prob /= np.maximum(cnt, 1)
            pred = (prob > 0.5) & wgt
            true = tgt & wgt
            maes.append(abs(pred.mean() - true.mean()))
            inter, union = (pred & true).sum(), (pred | true).sum()
            ious.append(inter / union if union else 1.0)
    model.train()
    return float(np.mean(maes)), float(np.mean(ious))


def save_checkpoint(model, args, history) -> None:
    import torch

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "arch": "unet",
        "encoder": args.encoder,
        "classes": 1,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "history": history,
    }, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", default="", help="директория с масками PNG от разметки")
    ap.add_argument("--prep-dir", default="reports/talc_train_data")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--skip-prepare", action="store_true")
    ap.add_argument("--out", default="models/talc_unet.pt")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--spi", type=int, default=8, help="кропов на снимок за эпоху")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.22)
    ap.add_argument("--n-neg", type=int, default=40)
    ap.add_argument("--dense", action="store_true",
                    help="маски уже уточнены (SAMRefiner): плотная разметка")
    ap.add_argument("--init-encoder", default="",
                    help="чекпоинт для инициализации (state_dict в нашем формате)")
    ap.add_argument("--freeze-encoder-epochs", type=int, default=0)
    ap.add_argument("--loss", default="gce", choices=["gce", "agce", "tloss"],
                    help="FG-терм слабого лосса")
    ap.add_argument("--agce-a", type=float, default=4.0,
                    help="параметр асимметрии AGCE (a>=0; больше = робастнее к шуму FG)")
    args = ap.parse_args()

    if not args.skip_prepare:
        assert args.masks, "--masks обязателен для подготовки данных"
        prepare_examples(Path(args.masks), Path(args.prep_dir), n_neg=args.n_neg,
                         dense=args.dense)
    if not args.prepare_only:
        train(args)


if __name__ == "__main__":
    main()
