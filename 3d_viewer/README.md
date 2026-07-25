# Lục địa 3D — viewer + tracker per-joint (Christina Cam-2)

Trực quan hoá tracking dưới dạng **lục địa 3D**: X = ngang equirect (0..4000), Y = cao (0..2000),
**Z = thời gian** (6fps). Mỗi khớp là một chấm; nối theo thời gian thành "sợi". Đắp được video
equirect nền để mắt kiểm track có hợp lý không.

Mở: phục vụ tĩnh thư mục này (vd `python3 -m http.server 8010`) rồi vào `christina_3d.html`.
Cần `det3d_*.json` (dẫn xuất, KHÔNG commit — xem .gitignore) + `frames6/` (JPEG 1/6s) cùng thư mục.

## Ba tracker (dropdown MODE) — cùng lõi Kalman per-joint, khác tầng NỐI

- **export3d_kalman.py** → Kalman · nối CẢ-THÂN: mỗi track ôm 17 khớp, so trung bình cả-thân,
  buộc 17 khớp theo 1 detection.
- **export3d_jointindep.py** → joint ĐỘC LẬP · mỗi LOẠI khớp = 1 track (đầu chỉ nối đầu, vai chỉ
  nối vai → không ráp bộ xương → KHÔNG chimera). **Có PHÂN NHÁNH**: hễ ≥2 số-đo trong gate 3σ
  → đẻ nhánh mỗi cái, nuôi song song; nhánh sai đoán trật → coast → tự chết. **Gia phả (PA)**
  giữ cha mỗi nhánh → click 1 chấm = cô lập CẢ CHUỖI sinh→tử.
- (greedy · IoU box = mode tham chiếu, không Kalman — script riêng.)

## Quyết định thiết kế (theo luật user)

- **Luật 12 (không ngưỡng bịa):** đã BỎ vạch "hoà" TIE=1.20 — đo thấy tỉ số cost #2/#1 TRƠN,
  không có khe. Giờ hễ ≥2 số-đo trong gate là rẽ. Số duy nhất còn lại = **GATE=3σ** (mức tin cậy
  thống kê, dẫn từ σ Kalman — có gốc).
- **Không cắt lén:** MAXH=80 chỉ là trần an toàn, log nếu chạm (thực tế chưa chạm).
- **Track hết, lọc sau (luật 6/10):** rẽ hết → ~82% track là stub (nhánh chết) → lọc bằng
  slider "% DÀI nhất" (xếp theo độ dài CHUỖI, không đoạn lẻ) và "track tối thiểu".
- **Click = màn hình:** bắt điểm theo pixel (không dùng raycaster ngưỡng thế-giới).

Schema điểm (jointindep): `[px,py,px,py, t, tid, conf, jt, vx,vy,sig, ppx,ppy]` + `pa{id:cha}`.
