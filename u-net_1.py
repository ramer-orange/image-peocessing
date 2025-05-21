import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from pathlib import Path
import random
import matplotlib.pyplot as plt
from medpy.metric.binary import hd95
from torch.utils.data import ConcatDataset, Subset
import requests
import json

def run_training():
    # ---------- ハイパーパラメータ ----------
    H, W       = 256, 256
    batch_size = 32
    num_epochs = 1
    lr         = 1e-3
    device     = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---------- transform ----------
    def mask_to_class(pil_img):
        arr = np.array(pil_img)
        cls = (arr > 0).astype(np.int64)  # 0→0, 255→1 にマッピング
        return torch.from_numpy(cls)    # torch.LongTensor (H, W)

    base_tf = transforms.Compose([
        transforms.Resize((H, W)),
        transforms.ToTensor(),
    ])

    # 水平反転
    flip_tf = transforms.Compose([
        transforms.Resize((H, W)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
    ])

    # 回転のみ（±10度までランダム）
    rotate_tf = transforms.Compose([
        transforms.Resize((H, W)),
        transforms.RandomAffine(degrees=10),     # -10〜+10度の範囲で回転
        transforms.ToTensor(),
    ])

    # 平行移動のみ（X方向に約10px移動）
    trans_tf_1 = transforms.Compose([
        transforms.Resize((H, W)),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.04, 0)
        ),
        transforms.ToTensor(),
    ])
    
        # 平行移動のみ（Y方向に最大限約10px移動）
    trans_tf_2 = transforms.Compose([
        transforms.Resize((H, W)),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.0, 0.04)
        ),
        transforms.ToTensor(),
    ])
    
            # 平行移動のみ（X方向、Y方向に最大限約10px移動）
    trans_tf_3 = transforms.Compose([
        transforms.Resize((H, W)),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.04, 0.04)
        ),
        transforms.ToTensor(),
    ])

    mask_tf = transforms.Compose([
        transforms.Resize((H, W), interpolation=Image.NEAREST),
        transforms.Lambda(mask_to_class),
    ])

    # --- Dataset ---
    class SegmentationDataset(Dataset):
        def __init__(self, images_dir, masks_dir, transform=None, target_transform=None):
            self.images = sorted(Path(images_dir).glob('*'))
            self.masks  = sorted(Path(masks_dir).glob('*'))
            self.transform        = transform
            self.target_transform = target_transform

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img = Image.open(self.images[idx]).convert('RGB')
            msk = Image.open(self.masks[idx])
            if self.transform:
                img = self.transform(img)
            if self.target_transform:
                msk = self.target_transform(msk)
            return img, msk

    # --- パス設定 ---
    base_path  = './dataset'  # データセットのルートディレクトリ
    images_dir = f'{base_path}/Images'
    masks_dir  = f'{base_path}/Ground-truths'

    # --- Dataset & DataLoader ---
    dataset = SegmentationDataset(
        images_dir=images_dir,
        masks_dir =masks_dir,
        transform=base_tf,
        target_transform=mask_tf
    )


    n       = len(dataset)
    n_train = int(n * 0.7)
    n_val   = int(n * 0.1)

    # インデックスをシャッフルして分割
    indices = list(range(n))
    torch.manual_seed(42)
    random.shuffle(indices)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    # インスタンス生成
    full_base = SegmentationDataset(
        images_dir, masks_dir,
        transform=base_tf,
        target_transform=mask_tf
    )
    aug_flip   = Subset(SegmentationDataset(images_dir, masks_dir, transform=flip_tf,   target_transform=mask_tf), train_idx)
    aug_rotate = Subset(SegmentationDataset(images_dir, masks_dir, transform=rotate_tf, target_transform=mask_tf), train_idx)
    aug_trans_1  = Subset(SegmentationDataset(images_dir, masks_dir, transform=trans_tf_1,  target_transform=mask_tf), train_idx)
    aug_trans_2  = Subset(SegmentationDataset(images_dir, masks_dir, transform=trans_tf_2,  target_transform=mask_tf), train_idx)
    aug_trans_3  = Subset(SegmentationDataset(images_dir, masks_dir, transform=trans_tf_3,  target_transform=mask_tf), train_idx)

    # 元の訓練データを生成
    train_base = Subset(full_base, train_idx)

    # 水平反転、回転、平行移動の訓練データと結合
    train_ds = ConcatDataset([train_base, aug_flip, aug_rotate, aug_trans_1, aug_trans_2, aug_trans_3])

    # 検証データとテストデータの切り出し
    val_ds  = Subset(full_base, val_idx)
    test_ds = Subset(full_base, test_idx)

    train_dl = DataLoader(train_ds,   batch_size=batch_size, shuffle=True,  drop_last=True)
    val_dl   = DataLoader(val_ds,     batch_size=batch_size, shuffle=False, drop_last=False)
    test_dl  = DataLoader(test_ds,    batch_size=batch_size, shuffle=False, drop_last=False)

    print("全データ数(n):", n);
    print("元の訓練データ数(n_train):",  n_train)
    print("最終的な訓練データ数(train_ds):", len(train_ds))

    # ---------- U-Net実装 ----------
    class UNet(nn.Module):
        def __init__(self, in_ch=3, out_ch=2):
            super().__init__()
            def C(i, o):
                return nn.Sequential(
                    nn.Conv2d(i, o, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(o, o, 3, padding=1), nn.ReLU(inplace=True))
            self.c1 = C(in_ch, 64);  self.p1 = nn.MaxPool2d(2)
            self.c2 = C(64, 128);    self.p2 = nn.MaxPool2d(2)
            self.c3 = C(128, 256);   self.p3 = nn.MaxPool2d(2)
            self.c4 = C(256, 512);   self.p4 = nn.MaxPool2d(2)
            self.b  = C(512, 1024)
            self.u4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
            self.c5 = C(1024, 512)
            self.u3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
            self.c6 = C(512, 256)
            self.u2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.c7 = C(256, 128)
            self.u1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.c8 = C(128, 64)
            self.out = nn.Conv2d(64, out_ch, 1)

        def forward(self, x):
            c1 = self.c1(x)
            c2 = self.c2(self.p1(c1))
            c3 = self.c3(self.p2(c2))
            c4 = self.c4(self.p3(c3))
            b  = self.b(self.p4(c4))
            c5 = self.c5(torch.cat([self.u4(b), c4], 1))
            c6 = self.c6(torch.cat([self.u3(c5), c3], 1))
            c7 = self.c7(torch.cat([self.u2(c6), c2], 1))
            c8 = self.c8(torch.cat([self.u1(c7), c1], 1))
            return self.out(c8)

    model     = UNet(in_ch=3, out_ch=2).to(device)
    opt       = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()


    # ---------- 学習 ----------
    train_hist, val_hist = [], []
    for ep in range(1, num_epochs + 1):
        # train
        model.train(); tl = 0
        for img, msk in train_dl:
            img, msk = img.to(device), msk.to(device)#.squeeze(1) # squeeze(1)はmask_to_classで次元を調整済みのため不要な場合があります
            opt.zero_grad()
            loss = criterion(model(img), msk)
            loss.backward(); opt.step()
            tl += loss.item() * img.size(0)
        train_hist.append(tl / len(train_dl.dataset))

        # val
        model.eval(); vl = 0
        with torch.no_grad():
            for img, msk in val_dl:
                img, msk = img.to(device), msk.to(device)#.squeeze(1) # 同上
                vl += criterion(model(img), msk).item() * img.size(0)
        val_hist.append(vl / len(val_dl.dataset))
        print(f'Epoch {ep:02}/{num_epochs}  Train {train_hist[-1]:.4f}  Val {val_hist[-1]:.4f}')

    # ---------- テスト ----------
    model.eval(); tl = 0
    with torch.no_grad():
        for img, msk in test_dl:
            img, msk = img.to(device), msk.to(device)#.squeeze(1) # 同上
            tl += criterion(model(img), msk).item() * img.size(0)
    print('Test loss:', tl / len(test_dl.dataset))


    # dice定義
    def dice_coef(y_true, y_pred):
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        return (2. * intersection + 1.) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1.)

    # テストデータ全体に対するDice係数rとHD95の算出
    dice_scores = []
    hd95_scores = []

    model.eval()
    with torch.no_grad():
        for imgs, masks in test_dl:
            imgs     = imgs.to(device)
            masks_np = masks.squeeze(1).cpu().numpy().astype(np.uint8)

            logits   = model(imgs)
            probs    = torch.softmax(logits, dim=1)[:,1]
            preds_np = (probs.cpu().numpy() > 0.5).astype(np.uint8)

            for pred, true in zip(preds_np, masks_np):
                dice_scores.append(dice_coef(true, pred))

                # HD95 は予測 or 真値にオブジェクトがないと計算不可
                if pred.sum() == 0 or true.sum() == 0:
                    hd95_scores.append(np.nan)
                else:
                    hd95_scores.append(
                        hd95(pred, true, voxelspacing=(1.0,1.0))
                    )

    # 平均を取る時は nan を除外
    avg_dice = float(np.nanmean(dice_scores))
    avg_hd95 = float(np.nanmean(hd95_scores))

    print(f'Dice係数の平均: {avg_dice:.4f}')
    print(f'HD95の平均: {avg_hd95:.4f}')

    # ---------- Loss vs Epoch グラフ ----------
    fig1 = plt.figure(figsize=(6,4))
    plt.plot(range(1, num_epochs+1), train_hist, '-o', label='Train')
    plt.plot(range(1, num_epochs+1), val_hist,   '-s', label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Loss');
    plt.grid(True); plt.legend()
    plt.title('Loss vs Epochs');
    plt.tight_layout();
    
    model.eval()
    n_show = 4

    orig_ds        = test_ds.dataset
    subset_indices = test_ds.indices
    img_paths      = orig_ds.images
    mask_paths     = orig_ds.masks

    fig2, axes = plt.subplots(n_show, 3, figsize=(11, 3 * n_show))

    for row, sub_idx in enumerate(random.sample(range(len(test_ds)), n_show)):
        # --- 元データセット上のインデックスと Path ---
        real_idx   = subset_indices[sub_idx]
        img_path   = img_paths[real_idx]
        mask_path  = mask_paths[real_idx]

        # --- 入力画像 (tensor→numpy) ---
        img_tensor, _   = test_ds[sub_idx] # img_tensorとして受け取る
        img_np   = img_tensor.permute(1, 2, 0).cpu().numpy() # img_tensorを使用

        # --- Ground-truth (0/255→0/1) ---
        msk_img  = Image.open(mask_path).convert('L').resize((W, H), Image.NEAREST)
        msk_np   = (np.array(msk_img) > 0).astype(np.uint8)

        # --- 推論 ---
        with torch.no_grad():
            pred_np = model(img_tensor.unsqueeze(0).to(device)).argmax(1).squeeze(0).cpu().numpy() # img_tensorを使用

        # --- 描画 ---
        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title(f'Input : {img_path.name}')
        axes[row, 1].imshow(msk_np, cmap='gray')
        axes[row, 1].set_title(f'GT    : {mask_path.name}')
        axes[row, 2].imshow(pred_np, cmap='gray')
        axes[row, 2].set_title('Prediction')

        for col in range(3):
            axes[row, col].axis('off')

    plt.tight_layout()
    
    return fig1, fig2

# ── Slack 通知用関数 ──
def post_to_slack(text: str, webhook_url: str):
    payload = {
        "text": text
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(webhook_url, data=json.dumps(payload), headers=headers)
    if not response.ok:
        print(f"Slack 送信エラー: {response.status_code} {response.text}")

SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/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"

# ── Slack へ通知 ──
if __name__ == "__main__":
    try:
        fig1, fig2 = run_training()
        # 正常終了したら通知
        post_to_slack(
            f"🎉 学習完了！\n",
            SLACK_WEBHOOK_URL
        )
    except Exception as e:
        # エラー発生時にも通知
        err_text = (
            f"❌ 学習中にエラー発生\n"
            f"Error: {type(e).__name__}: {e}"
        )
        try:
            post_to_slack(err_text, SLACK_WEBHOOK_URL)
        except Exception:
            # 通知すら失敗したら、ここでログに出すだけ
            print("Slack 通知失敗:", err_text)
        # エラーを上位に伝搬（ターミナルにもトレースバックを出す）
        raise
    
    # グラフを表示
    fig1.show()
    fig2.show()
    input("グラフを確認したら Enter を押してください…")