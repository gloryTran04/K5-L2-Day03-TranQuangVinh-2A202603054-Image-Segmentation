# Báo cáo thực hành Day 3 — Image Segmentation

- Mã học viên theo lớp: 2A202603054
- Ngày / CVAT local: 11–12/11/2026, CVAT local v2.82.0 (Docker)
- Công cụ đã dùng: CVAT local; SAM3
## 1. Bài đã nộp

| Task | File ZIP đúng tên | Hoàn thành mấy ảnh | Điểm tối đa (coach chấm sau) |
| --- | --- | ---: | ---: |
| easy_semantic | `submissions/easy_semantic.zip` (Segmentation mask 1.1) | 3 / 3 | 20 |
| medium_instance | `submissions/medium_instance.zip` (COCO 1.0) | 3 / 3 | 32 |
| hard_panoptic | `submissions/hard_panoptic.zip` (COCO 1.0) | 2 / 2 | 30 |
| cp1_holes | `submissions/cp1_holes.zip` (COCO 1.0) | 1 / 1 | 3 |
| cp2_slice | `submissions/cp2_slice.zip` (COCO 1.0) | 1 / 1 | 3 |
| cp5_occlusion | `submissions/cp5_occlusion.zip` (COCO 1.0) | 1 / 1 | 3 |
| cp3_thin | `submissions/cp3_thin.zip` (Segmentation mask 1.1) | 1 / 1 | 3 |
| cp4_curb | `submissions/cp4_curb.zip` (Segmentation mask 1.1) | 1 / 1 | 3 |
| cp6_coverage | `submissions/cp6_coverage.zip` (Segmentation mask 1.1) | 1 / 1 | 3 |
| **Tổng tối đa** | | | **100** |

Cả 9 ZIP qua `scripts/inspect_submissions.py` với trạng thái `OK`. Số object đã nộp: Medium 77 (ảnh `000000181542`: 22, `000000373353`: 30, `000000458325`: 25), Hard 79 segment (stuff + từng thing), cp1 5, cp2 14, cp5 28. Không có export lỗi.

## 2. Một quyết định trước khi dùng gợi ý

- Ảnh, vị trí và object Medium đầu tiên tự vẽ: ảnh `000000181542.jpg`, người phụ nữ mặc áo dài đi giữa vạch qua đường (khoảng x 188–305, y 131–460). Vẽ bằng Polygon trên CVAT, 35 điểm, trước khi bật bất kỳ gợi ý nào; object này giữ nguyên trong bài nộp, không cho gợi ý chỉnh lại.
- Class và quy tắc tôi dùng để chọn biên: `person`. Bám theo phần nhìn thấy: tóc, hai tay, vạt áo dài và hai bàn chân; không lấy bóng trên mặt đường, không lấy phần xe máy phía sau dù xe nằm sát vạt áo. Chỗ vạt áo trắng lẫn với vạch trắng ở dưới, tôi dừng ở mép vải (có nếp gấp và bóng), không kéo theo vạch sơn.
- Nếu dùng gợi ý sau đó: vùng gợi ý sai/đúng, hành động sửa/giữ và lý do:
  - Với các object còn lại, tôi vẽ polygon trước rồi lấy khung của chính polygon đó làm prompt cho SAM. Chỉ giữ mask của SAM khi nó trùng với polygon của mình (IoU ≥ 0,6, diện tích không lệch quá); 10 object ở Medium không đạt điều kiện này nên giữ nguyên polygon tay. Sau đó, object nào có mask Mask2Former COCO cùng class và trùng ≥ 0,6 với nhãn của tôi thì dùng mask đó; ở Hard cách này làm điểm giảm nên tôi không dùng cho thing của Hard.
  - Mask2Former COCO đề xuất một `motorcycle` ở ảnh `000000458325.jpg` (khoảng x 232–255, y 214–237). Phóng to thì đó là xe đẩy em bé của nhóm người đi bộ, không phải xe máy và không có trong class list, nên tôi không thêm.
  - Ở các task semantic, gợi ý của Mask2Former hay tô nắp capo và táp-lô xe mình thành `car` (cp6) hoặc `sidewalk` (cp4). Tôi bỏ toàn bộ vùng đó, để trống.
  - Ở ảnh cao tốc `7ee6d192-89e2408b.jpg`, gợi ý tô dải tường chắn bê tông bên trái thành `sidewalk` và hai đốm nhỏ trên sườn đồi thành `building`. Đường cao tốc không có vỉa hè đi bộ và ở đó cũng không có nhà, nên tôi xóa hai nhãn này, để trống tường chắn.

## 3. Một lỗi tôi tìm thấy và sửa

- Task/ảnh/vùng: `medium_instance`, chủ yếu ảnh `000000458325.jpg` (hàng xe đỗ hai bên phố) và nhóm người nhỏ ở hậu cảnh của cả ba ảnh (ví dụ dãy người dưới đuôi xe bán tải ở `000000181542.jpg`, x 270–372, y 114–148).
- Lỗi thuộc loại: thiếu vật + biên (mask quá thô).
- Bằng chứng tôi nhìn thấy: lần chấm đầu Medium chỉ đạt metric 0,385 (mean matched IoU 0,70 × recall 0,55), tức 0/32. Mở lại CVAT thì thấy khoảng 15 người nhỏ ở xa chưa có mask, còn polygon xe đỗ gần như hình chữ nhật, ăn sang xe bên cạnh ở mép trước và hụt mép sau, nên nhiều xe chỉ đạt IoU 0,40–0,49, dưới ngưỡng 0,5.
- Quy tắc và hành động sửa: mỗi vật một mask, bám phần nhìn thấy. Tôi vẽ lại hàng xe đỗ theo đúng thứ tự xe trước che xe sau (xe sau chỉ còn phần nóc và hông lộ ra), thêm từng người nhỏ ở hậu cảnh, sau đó dùng SAM với khung từ polygon của mình để làm sát biên, rồi xem lại ảnh chồng mask của từng ảnh trước khi Save.
- Sau sửa đã Save và export lại chưa? Đã Save và export lại `medium_instance.zip` (COCO 1.0), chạy lại `inspect_submissions.py`: OK, 77 annotation.

Kết quả tự đánh giá liên quan: Medium 0,385 → 0,584 sau khi vẽ lại tay → 0,691 sau khi làm sát biên (0 → 20,7/32). Tổng ba tier: **20,2/82 ở bản làm trước khi xem reference**, **63,6/82 ở bản nộp** (Easy 19,6 · Medium 20,7 · Hard 23,3) sau khi sửa theo phản hồi; chi tiết trong `reports/tiers/SCORECARD.md`. Phần tăng này có được sau khi reference đã phát, tôi tách riêng như yêu cầu. Không đưa file ground truth vào fork.

## 4. Ba ca chưa chắc hoặc đã cân nhắc

| Ảnh/vị trí | Hai cách hiểu có thể | Quy tắc/chứng cứ | Quyết định hoặc câu hỏi cho coach |
| --- | --- | --- | --- |
| 1. `hard_panoptic` / `000000460147.jpg`, lòng đường bên trái nơi ô tô trắng đang chạy (x ~0–330, y ~150–424) | `road` hay `sidewalk` | Có vạch kẻ làn, xe đang chạy, bó vỉa ngăn với lề trái. Theo quy tắc ranh theo chức năng thì đây là đường xe chạy. Khi tự đánh giá, reference lại tính vùng này là sidewalk | Giữ `road`. Câu hỏi: reference gán sidewalk cho lòng đường này là cố ý hay nhầm? Nếu cố ý thì quy tắc nào ạ? |
| 2. `hard_panoptic` / `000000460147.jpg`, xe chở ô tô giữa đường (x ~316–437, y ~185–368) | Một `truck` gồm luôn các ô tô trên sàn, hay `truck` + từng `car` trên sàn là instance riêng | Các ô tô trên sàn nhìn rõ từng chiếc, có biên riêng; xe chở là vật chở hàng | Gán xe chở là `truck`, từng ô tô trên sàn là `car` riêng. Câu hỏi: hàng hóa là xe trên xe tải có nên tách instance không? |
| 3. `cp4_curb` / `7d83710e-4697c3b2.jpg`, bồn đất quanh gốc cây giữa lòng đường và vỉa hè (x ~510–885, y ~335–545) | Gộp vào `sidewalk` hay để trống | Bồn đất có bó vỉa riêng, không dùng để đi bộ, task chỉ có `road` và `sidewalk`; vùng không thuộc hai lớp thì không ép | Để trống. Ranh `road`–`sidewalk` lấy theo mép bó vỉa phía đường, không theo màu bê tông |
