# Lane Detection Ugot Robot using Opencv

## 10/08/2024
**Line Detection sử dụng HoughLines OpenCV vẽ viền và vẽ các điểm giữa của line**
 ./LaneDetection.py
- Task
  - Detect được line
  - Tìm khoảng cách từ điểm giữa line đến trung tâm frame
- Problem
  - Chưa tìm được chính xác điểm giữa của line
  - Bị nhiễn điểm giữa nhiều
- Next Step
  - Chỉnh sửa độ nhiễu
  - Kiểm tra lại công thức tìm điểm giữa
  
## 13/08/2024
**Thay đổi cách Detect Line**
/linev2.py
- Task
  - Detect Line
  - Vẽ viền line
- Problem:
  - Chưa nhận diện rõ các ngã 3, ngã 4
- Next Step:
  - Line follow theo line đã được detect