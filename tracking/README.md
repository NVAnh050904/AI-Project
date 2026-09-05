# Tracking Module (Level 1 — Baseline)

Module này tách biệt hoàn toàn với `models/hydraplus` (Attribute Recognition).
Mục tiêu: có một pipeline **YOLOv8 (detect person) + ByteTrack (track)** chạy
ổn định trước, output ra dữ liệu structured để các module sau (Attribute,
Re-ID) đọc lại mà không phải sửa kiến trúc.

## Cài đặt thêm (chưa có trong requirements.txt hiện tại)

```powershell
pip install ultralytics opencv-python
```

`ultralytics` đã tích hợp sẵn ByteTrack và BoT-SORT (`bytetrack.yaml`,
`botsort.yaml`), không cần cài/tự viết tracker riêng.

## 1. Chạy tracking baseline

```powershell
# Video file
python tracking/track.py --source path/to/video.mp4

# Webcam (test nhanh khi chưa có video mẫu)
python tracking/track.py --source 0 --show

# Lưu lại video có vẽ bbox + track_id để kiểm tra bằng mắt
python tracking/track.py --source video.mp4 --save-video
```

Output:
```text
reports/tracking/<ten_video>_tracks.csv
frame_id,track_id,x1,y1,x2,y2,confidence,timestamp
1,1,120.5,80.2,250.1,480.9,0.91,0.04
1,2,500.0,100.3,620.4,490.1,0.87,0.04
```

`yolov8n.pt` sẽ tự động tải về (pretrained COCO) trong lần chạy đầu tiên —
đây chỉ là detector person tạm dùng ở Level 1, chưa cần fine-tune.

## 2. Đánh giá (khi có ground-truth / video benchmark)

Chưa viết script này (cần dataset có nhãn track chuẩn, ví dụ MOT-format)
— sẽ thêm khi bạn có dữ liệu benchmark cụ thể. Các chỉ số cần theo dõi:
IDF1, MOTA, HOTA, ID switches (xem thảo luận roadmap trước đó).

## 3. Trích crop theo track_id (chuẩn bị nối sang Attribute Recognition)

```powershell
# Extract crops từ CSV tracking (dùng --clean để làm sạch crop cũ)
python tracking/extract_crops.py --video path/to/video.mp4 --csv reports/tracking/video_tracks.csv --output-dir reports/tracking/crops --every-n-frames 5 --clean
```

Kết quả: `reports/tracking/crops/track_<id>/frame_<n>.jpg`.

## 4. Gán & Tổng hợp thuộc tính UPAR theo Track ID (`track_attributes.py`)

```powershell
python tracking/track_attributes.py \
  --crops-dir reports/tracking/crops \
  --tracks-csv reports/tracking/real_pedestrians_tracks.csv \
  --checkpoint checkpoints/hydraplus_upar_best.pth \
  --output-dir reports/tracking \
  --min-frames 3
```

Output:
- `reports/tracking/track_attributes.csv`: Bảng thuộc tính Top-1 từng track.
- `reports/tracking/track_attributes.json`: Chi tiết multi-label và 40 xác suất raw.
- `reports/tracking/tracked_persons_summary.csv`: Bảng tổng hợp đối tượng (Metadata tracking + Attributes).

## Chưa làm ở bước này (cố tình)

- Không train tracker.
- Không train detector riêng (dùng pretrained COCO `yolov8n.pt` trước,
  benchmark lên `yolov8s.pt` nếu person nhỏ/occlusion nhiều).
- Chưa có Re-ID embedding — đó là Level 3, chỉ thêm sau khi tracking Level 1
  chạy ổn định và ID switches ở mức chấp nhận được.
