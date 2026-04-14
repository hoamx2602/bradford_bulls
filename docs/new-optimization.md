# Frame Reconstruction Pipeline — Nghien cuu & Phat trien

## 1. Van de goc

### Boi canh
Pipeline hien tai (`frame_extraction`) da co kha nang loc va chon ra cac frame "net nhat" tu video tran dau (30 FPS, 1080p). Tuy nhien, ngay ca nhung frame duoc chon la net nhat van bi **motion blur** — do cau thu di chuyen nhanh trong video the thao.

### Quan sat quan trong (tu nguoi dung)
> Khi xem video truc tiep, mat nguoi co the nhin ro logo tren ao cau thu. Nhung khi dung lai bat ky frame nao, logo deu bi mo. Nao bo tich hop thong tin tu nhieu frame lien tiep de "nhin thay" hinh anh net — mot hinh anh ma thuc te khong ton tai trong bat ky frame don le nao.

### He qua
- Cac frame hien tai **kho annotate bang mat nguoi** vi logo khong du net
- Model train tren data mo se co accuracy thap hon
- Can mot cach de tao ra frame NET THUC SU tu video

---

## 2. Y tuong cot loi

**Khong inference truc tiep video vao model.** Thay vao do:

1. **Preprocessing**: Tu video, trich xuat cac frame net (pipeline hien tai)
2. **Frame Reconstruction**: Tai moi frame da chon, dung cac frame lan can de **tao ra 1 frame MOI hoan toan net** — frame nay khong ton tai trong video goc
3. **Annotation/Inference**: Dung frame moi nay de annotate hoac chay model detect logo

Pipeline nay ap dung cho **ca training data va inference tren video moi** — thong nhat 1 preprocessing pipeline duy nhat.

### Loi ich
- **Annotation de hon**: Logo ro rang, annotator nhin thay chinh xac
- **Model accuracy cao hon**: Train tren data net → detect tot hon
- **Giam compute inference**: Chi chay model tren vai nghin frame net thay vi 162K frames (30 FPS x 90 phut)
- **Khong can real-time**: Video moi cung duoc preprocessing truoc
- **Pipeline thong nhat**: Cung 1 pipeline cho ca training va inference

---

## 3. Cac giai phap da thu nghiem

### 3.1 Multi-frame fusion v1 — Global alignment + Median (KHONG HIEU QUA)

**Cach lam:**
- Lay ±5 frames quanh frame A
- Align tat ca frames ve frame A bang optical flow (Farneback)
- Median fusion toan bo frame

**Ket qua:**
- Frame 462 (cau thu it di chuyen): Sac net hon o vung background, logo "KLG" ro hon 1 chut
- Frame 835 (cau thu di chuyen): **Ghosting nang** — moi cau thu di chuyen theo huong khac nhau, optical flow khong align duoc

**Nguyen nhan that bai:** Global optical flow chi xu ly duoc camera motion (panning), khong xu ly duoc local motion cua tung cau thu.

### 3.2 Multi-frame fusion v2 — Player-level tracking + Fusion (COI TIEN NHUNG CHUA DU)

**Cach lam:**
- YOLO detect cau thu trong frame A
- Template matching track tung cau thu qua cac frame lan can
- Align va fuse chi vung player crop
- Paste lai len base frame

**Ket qua:**
- Sharpness tang 55-79% cho tung cau thu khi it di chuyen
- Frame 462: Logo "KLG" net hon, "SPORTS EVENTS SERVICE" doc duoc
- Frame 835: Van co ghosting o cau thu chay nhanh, 1 player sharpness giam -23%

**Nguyen nhan han che:** Template matching khong du chinh xac o pixel level. Cau thu chay nhanh van bi ghosting. Edge blending giua player crop va background co seam.

### 3.3 Temporal Focus Stacking v3 — Local sharpness selection (THAT BAI)

**Y tuong:** Giong nao bo — tai moi pixel, chon version net nhat tu tat ca cac frame.

**Cach lam:**
- Align frames bang optical flow
- Tinh local sharpness (Laplacian) tai moi pixel cho moi frame
- Moi pixel duoc lay tu frame nao co sharpness cao nhat tai vi tri do
- Blend bang Laplacian pyramid

**Ket qua:** Artifact va ghosting **nang nhat** trong tat ca cac phuong phap. Sharpness score tang (587 → 1132) nhung do la **canh gia** tu artifact, khong phai detail that.

**Nguyen nhan that bai:** Optical flow alignment khong du chinh xac o pixel level → weight map bi noisy → moi pixel chon tu frame khac nhau nhung alignment lech → tao ra artifact.

### 3.4 NAFNet single-image deblurring (TOT NHUNG CHUA DUNG Y TUONG)

**Cach lam:**
- Dung NAFNet (pre-trained tren GoPro dataset) deblur tung frame rieng le
- Tiling 256x256, chay tren Apple Silicon MPS
- Khong dung thong tin tu frame lan can

**Ket qua:**
- Frame 462: Sharpness 121.7 → 441.0 (3.6x), logo net hon, khong artifact
- Frame 835: Sharpness 133.2 → 4095.8 (30.8x), nhung **artifact o vung broadcast overlay** (BullsTV logo bi bien dang)

**Danh gia:** Tot nhat trong cac phuong phap da thu, nhung **chi dung 1 frame** — chua tan dung thong tin tu cac frame lan can nhu y tuong goc.

---

## 4. Giai phap duoc chon: Video Restoration Model (BasicVSR++)

### 4.1 Tai sao BasicVSR++?

Tat ca cac phuong phap tren deu co chung 1 han che: **khong hieu motion**. Chung co gang align pixel bang optical flow roi tron, nhung video the thao co motion qua phuc tap (nhieu nguoi, nhieu huong, nhieu toc do).

BasicVSR++ (CVPR 2022) giai quyet dung bai toan nay:
- **Input**: chuoi N frames lien tiep
- **Output**: 1 frame MOI duoc reconstruct, net hoan toan
- **Cach hoat dong**: Bidirectional propagation — model "nhin" ca frame truoc va sau, tu hoc optical flow ben trong, trich xuat thong tin tu moi frame, roi tong hop thanh 1 frame moi
- **Khong can alignment thu cong** — model tu hoc
- **Da train tren video thuc te** voi motion blur

### 4.2 Cac model tuong tu

| Model | Nam | Uu diem | Nhuoc diem |
|-------|-----|---------|------------|
| **BasicVSR++** | CVPR 2022 | Can bang performance/compute, pretrained san | - |
| RVRT | NeurIPS 2022 | Chat luong cao hon (Transformer) | Nang hon |
| VRT | ICCV 2023 | State-of-the-art | Rat nang, kho chay |

**Chon BasicVSR++ lam starting point** — can bang giua chat luong va kha nang chay tren Colab.

### 4.3 Pipeline hoan chinh

```
1. Pipeline hien tai (Pass 1 + Pass 2)
      → Frame A (timestamp tot, co cau thu target, co zoom du)

2. Mo cua so ±15 frames quanh A (khoang 1 giay o 30 FPS)

3. Scene change detection (histogram diff)
      → Trim cua so tai scene boundary gan nhat
      → Dam bao tat ca frames trong cua so thuoc cung 1 scene

4. IF cua so >= 7 frames:
      → BasicVSR++ reconstruct → Frame A' (net, moi hoan toan)
   ELSE (cua so qua hep, frame A nam sat scene cut):
      → NAFNet single-frame deblur → Frame A' (fallback)

5. Output Frame A'
      → Dung de annotate (Phase 1)
      → Hoac dung de inference detect logo (Phase 3)
```

---

## 5. Scene Change Detection

### 5.1 Tai sao can?

Khi lay cac frame lan can cua frame A, co the bi lay nham frame thuoc scene khac (camera cut, replay, graphics). Model nhan input "lan lon" → output se rac.

### 5.2 Phuong phap: Histogram Difference

```
Frame 450 → 451: diff = 0.02 (cung scene)
Frame 454 → 455: diff = 0.85 (CAMERA CUT)
Frame 455 → 456: diff = 0.03 (cung scene moi)
```

- Tinh color histogram cua moi frame
- So sanh histogram giua 2 frame lien tiep (chi-square hoac correlation)
- Camera cut → histogram thay doi dot ngot → de detect

### 5.3 Cac truong hop dac biet

**Replay:**
- Camera cut di replay roi cut lai
- Histogram diff phat hien duoc ca 2 lan cut
- Frame A co the nam trong replay hoac ngay sau replay

**Slow-motion replay:**
- Khong co hard cut, nhung FPS khac → motion pattern khac
- Co the detect bang: broadcast overlay thay doi (scoreboard bien mat khi replay)

**Camera zoom nhanh:**
- Khong phai scene change nhung histogram thay doi lon → false positive
- Giai phap: threshold du cao (zoom ~0.3-0.4, cut ~0.7+)
- Hoac ket hop SSIM: SSIM thap = scene khac, SSIM trung binh = cung scene nhung zoom

### 5.4 Edge case: Cua so qua hep

Neu frame A nam sat scene boundary (chi 3 frames cung scene 1 phia):
- **Option 1**: Asymmetric window — lay it frame truoc, nhieu frame sau (BasicVSR++ van hoat dong)
- **Option 2**: Neu < 7 frames → fallback sang NAFNet single-frame deblur

---

## 6. Generative "lam net" vs Reconstruction — Phan biet quan trong

### 6.1 Van de phat hien tu thu nghiem

Khi dung ChatGPT hoac Gemini de "lam net" anh mo, ket qua thoat nhin co ve net, nhung:
- Logo mo → model **tuong tuong** ra logo khac hoan toan
- Detail duoc **hallucinate** — tao pixel moi tu knowledge cua model, khong phai tu du lieu that
- Mot so anh thu nghiem: logo net nhung **SAI HOAN TOAN** so voi logo that

### 6.2 Hai loai "lam net"

| | Generative (ChatGPT/Gemini) | Reconstruction (RVRT/BasicVSR++) |
|---|---|---|
| Nguon thong tin | Knowledge tu training data | Pixel that tu video goc |
| Cach hoat dong | "Doan" detail → hallucinate | Tong hop info tu nhieu frame → reconstruct |
| Logo mo → | Tao logo moi (co the sai) | Ghep info that tu cac frame → dung |
| Tin cay cho annotation? | **KHONG** — data sai tu goc | **CO** — moi pixel co nguon goc that |
| Tao detail khong ton tai? | Co (hallucination) | Khong |

### 6.3 Ket luan

**Generative approach KHONG phu hop** cho project nay vi:
- Annotation data sai → model train sai → exposure measurement sai → toan bo ket qua khong dang tin
- Khong co cach kiem chung logo nao la that, logo nao la hallucinated

**Reconstruction approach (RVRT/BasicVSR++)** la huong dung vi:
- Chi tong hop thong tin **da co san** trong cac frame lan can
- Khong tao ra detail moi — neu thong tin khong co trong bat ky frame nao thi output se mo (trung thuc)
- Frame reconstructed co the khong net bang generative output, nhung **dang tin 100%**
- Neu van chua du net → do la gioi han that cua thong tin trong video → can video chat luong cao hon (4K, 60 FPS)

---

## 7. Cac cau hoi da thao luan

### Q: Model train tren video thuong co hoat dong tot cho video the thao?
**A:** BasicVSR++ train tren REDS/Vimeo-90K (scene thuong: nguoi di bo, xe co). Video the thao co motion nhanh hon. **Kha nang cao van hoat dong** vi motion blur pattern tuong tu, nhung co the khong perfect — can test thuc te. Neu khong du tot, co the fine-tune tren data Bradford Bulls.

### Q: Can bao nhieu frames lan can?
**A:** BasicVSR++ thuong dung cua so 15-30 frames. Voi 30 FPS, ±7 frames = ~500ms — du context. Nhieu hon khong nhat thiet tot hon vi scene co the thay doi (camera cut).

### Q: Compute co chay duoc tren MacBook?
**A:** BasicVSR++ voi 1920x1080 se nang. **Giai phap**: chay tren Google Colab GPU (da co trong tech stack, pipeline hien tai cung chay tren Colab). Cach trien khai tuong tu file `02_team_aware_extraction.ipynb`.

### Q: Co can fine-tune model khong?
**A:** Pretrained weights co the du tot cho muc dich annotation. Neu ket qua chua on, co the fine-tune — nhung do la buoc sau.

### Q: Tai sao khong inference truc tiep video vao model (PLAN cu: 30 FPS full video)?
**A:** 
- 90 phut x 30 FPS = ~162,000 frames → compute khong lo
- Nhieu frame mo → false positive cao, confidence khong dang tin
- Pipeline moi: preprocessing → vai nghin frame net → inference nhanh 30-50x
- Relative comparison giua cac logo (muc tieu chinh) khong bi anh huong boi viec khong co moi frame

### Q: Cach tinh exposure time khi khong co moi frame?
**A:** 
- **Segment-based estimation**: Trong 1 segment 5 giay, extract 10 frame net, 8/10 co logo A → uoc tinh logo A xuat hien ~80% x 5s = 4s
- **Interpolation**: Logo xuat hien o frame 100 va frame 130 → hop ly suy ra logo xuat hien lien tuc trong khoang do
- Weighted exposure (confidence x duration) van tinh duoc

---

## 8. Lua chon model cuoi cung: RVRT

### 8.1 Tai sao RVRT thay vi BasicVSR++?

Sau khi nghien cuu ky, RVRT (Recurrent Video Restoration Transformer, NeurIPS 2022) la lua chon tot hon cho muc tieu do chinh xac cao nhat:

| | BasicVSR++ | RVRT |
|---|---|---|
| Architecture | CNN + flow-guided deformable alignment | Transformer + guided deformable attention |
| Video deblurring | Mo rong tu SR, khong co pretrained deblur chinh thuc trong mmagic | **Co pretrained deblurring chinh thuc** (GoPro, 16 frames) |
| Chat luong | Tot | **State-of-the-art** (cao hon BasicVSR++) |
| Compute | Nhe hon | Nang hon, nhung chap nhan duoc tren T4 GPU |
| Inference | Phuc tap (can mmagic framework) | Don gian (standalone script) |

### 8.2 RVRT pretrained models

- `005_RVRT_videodeblurring_GoPro_16frames` — Train tren GoPro dataset, input 16 frames
- Inference: `--tile 0 256 256 --tile_overlap 2 20 20` (tiling de fit T4 GPU memory)
- Download tu: https://github.com/JingyunLiang/RVRT/releases

### 8.3 Pipeline cuoi cung

```
1. Pipeline hien tai (Pass 1 + Pass 2)
      → Frame A + metadata (timestamp, category, score...)

2. Doc metadata CSV → lay danh sach frame_num can reconstruct

3. Voi moi frame A:
   a. Mo cua so ±8 frames (16 frames tong cong, khop voi RVRT input)
   b. Scene change detection (histogram diff) → trim cua so
   c. IF cua so >= 7 frames:
        → RVRT reconstruct → Frame A' (net, moi hoan toan)
      ELSE:
        → NAFNet single-frame deblur → Frame A' (fallback)

4. Luu Frame A' thay the frame goc
5. Cap nhat metadata CSV (danh dau frame da duoc reconstruct)
```

---

## 9. Tong ket va Buoc tiep theo

### Da hoan thanh
- Pipeline frame extraction hoan chinh (Pass 1 + Pass 2)
- Team-aware classification (target vs opponent)
- Overlay mask de xu ly broadcast graphics
- Notebook extraction: `02_team_aware_extraction.ipynb`
- **Script reconstruction: `run_reconstruction.py`** (MOI)
  - RVRT video deblurring (16-frame input, GoPro pretrained)
  - NAFNet single-frame deblur (fallback)
  - Scene change detection (histogram diff)
  - Visual comparison + sharpness metrics
  - Roboflow upload

### Workflow tren Google Colab
```
Step 1: Chay 02_team_aware_extraction.ipynb
        → Output: frames/ + metadata CSV

Step 2: Chay run_reconstruction.py (truc tiep tren Colab terminal)
        → Input: metadata CSV + video goc
        → Output: frames_reconstructed/ (frame net) + reconstruction CSV
        → Test:   python run_reconstruction.py --video ... --csv ... --test
        → Prod:   python run_reconstruction.py --video ... --csv ...

Step 3: Upload frames_reconstructed/ len Roboflow de annotate
```

### Can lam tiep (sau khi test)
1. **Chay test mode tren Colab**: 10 frames, xem ket qua visual
2. **Danh gia chat luong**: Logo co doc duoc khong? Annotator co de lam khong?
3. **Neu tot**: chay production mode cho toan bo frames
4. **Neu chua tot**: xem xet fine-tune RVRT hoac dung video 4K/60FPS
5. **Tich hop vao Phase 3**: Dung frame reconstructed cho inference (detect logo + tinh exposure)

### Files prototype da tao (tham khao, khong dung trong production)
- `prototype_fusion.py` — v1: global alignment + median (khong hieu qua)
- `prototype_fusion_v2.py` — v2: player-level tracking + fusion (han che)
- `prototype_fusion_v3.py` — v3: temporal focus stacking (that bai)
- `prototype_deblur.py` — NAFNet single-image deblur (tot, dung lam fallback)

### Output mau da tao
- `output/fusion_test/` — Ket qua v1
- `output/fusion_v2/` — Ket qua v2
- `output/fusion_v3/` — Ket qua v3
- `output/deblur/` — Ket qua NAFNet
