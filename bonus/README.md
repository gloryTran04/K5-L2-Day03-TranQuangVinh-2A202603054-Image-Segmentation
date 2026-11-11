# Bonus — SAM 3 tự gán nhãn, import vào CVAT

Tool dùng **SAM 3** (Meta, segmentation theo câu lệnh chữ) để tạo nhãn nháp cho cả 9 task của Day 3. Kết quả được xuất **đúng định dạng CVAT export**, nên import thẳng vào CVAT để người gán nhãn chỉ việc sửa. Thư mục này tách riêng khỏi bài chính: `submissions/` vẫn là bài tự gán và kiểm tay.

## Cách hoạt động

```
classes.json của task ──► prompt chữ cho từng class (PROMPTS)
        │
ảnh ──► SAM 3 (Colab GPU) ──► mask + điểm tin cậy cho mỗi prompt
        │
        ├─ semantic : argmax điểm theo pixel; prompt "chặn" (car, grass, wall…) giữ pixel ngoài schema ở trạng thái unlabeled
        ├─ instance : mỗi vật 1 mask, bỏ trùng (IoU) giữa các prompt/class
        └─ panoptic : stuff gộp 1 vùng/class + từng thing; thing vẽ đè stuff → không chồng lấn
        │
        ▼
<task>.zip  (Segmentation mask 1.1 / COCO 1.0, cấu trúc y hệt CVAT export)
        │
CVAT: Task → Actions → Upload annotations → người sửa → Save → Export
```

| File | Vai trò |
| --- | --- |
| `sam3_autolabel.py` | Lõi: gọi SAM 3, hậu xử lý theo loại task, ghi ZIP định dạng CVAT, ảnh xem trước |
| `sam3_autolabel_colab.ipynb` | Chạy toàn bộ trên Google Colab (GPU T4) |
| `compare.py` | Đo mức tool đã làm được so với bài cuối (dùng metric của lab: mIoU / matched IoU × recall / PQ) |
| `test_export.py` | Test offline không cần GPU: runner giả trả về mask có sẵn, kiểm ZIP qua `inspect_submissions.py` và độ khớp ~100% |

## Chạy

1. Hugging Face: đồng ý license ở trang `facebook/sam3`, tạo access token (quyền *read*).
2. Upload `sam3_autolabel_colab.ipynb` lên Colab → Runtime **T4 GPU** → chạy từ BƯỚC 0 đến BƯỚC 8.
3. Giải nén `bonus_out.zip` tải về. Trên CVAT local, với mỗi task tạo **task mới tên `<mã_task>_sam3`** (cùng ảnh, cùng `cvat-labels.json`), rồi **Actions → Upload annotations**:
   - `easy_semantic`, `cp3_thin`, `cp4_curb`, `cp6_coverage` → format **Segmentation mask 1.1**
   - `medium_instance`, `hard_panoptic`, `cp1_holes`, `cp2_slice`, `cp5_occlusion` → format **COCO 1.0**
4. Sửa nhãn trong CVAT, ghi lại tool sai ở đâu.

Test không cần GPU (chỉ cần numpy, pillow, pycocotools):

```bash
python3 bonus/test_export.py --out /tmp/bonus_test_out
```

## Đo "tool làm được bao nhiêu %"

```bash
python3 bonus/compare.py --tool bonus_out --final submissions
```

- **agreement** = điểm của nhãn tự động khi lấy bài cuối (đã kiểm tay) làm chuẩn, theo đúng metric của lab. Đây là độ giống, chưa phải độ đúng.
- Khi có ground truth: chạy `scoring/score.py` cho cả `bonus_out/` và `submissions/` để so **tool vs đáp án** và **bài tay vs đáp án** (BƯỚC 7 trong notebook).

## Kết quả

| Task | Agreement tool vs bài cuối | Ghi chú lỗi điển hình của tool |
| --- | ---: | --- |
| (điền sau khi chạy BƯỚC 6) | | |

## Giới hạn đã biết

- SAM 3 không biết quy ước Cityscapes (cỏ = terrain, không phải vegetation), nên phải dùng prompt chặn và ngưỡng điểm. Ranh road/sidewalk theo bó vỉa vẫn hay sai và cần sửa tay.
- Class nhập nhằng (van/pickup = car hay truck, ô tô trên xe chở ô tô) do prompt quyết định, nên người gán nhãn phải kiểm.
- Mảnh vật dưới 30 pixel bị bỏ như nhiễu (`MIN_AREA`).
- Checkpoint `facebook/sam3` có giới hạn truy cập; token chỉ nhập lúc chạy và không lưu vào repo.
