Dữ liệu:
  - có 46 class,  mỗi class có khoảng 1400 ảnh
  - generate (file generate data): mỗi ảnh sẽ được tạo thành 12 ảnh bao gồm: 
    + 2 ảnh xoay: random từ -45 đến 45
    + zoom:  (ảnh gốc + 2 ảnh xoay) * 3 = 9 ảnh; ảnh zoom random từ 0.75 đến 1 và zoom theo trung tâm
    + tổng = ảnh gốc + 2 ảnh xoay + 9 ảnh zoom = 12 ảnh
  - Dữ liệu được nén vào file có tên train_data.file (12gb)
  - link
Model: (file train)
  - model theo bài báo: M6-1: conv3-32 -> conv3-32 ->maxpool -> conv3-64 -> conv3-64 -> maxpool ->flatten -> fc-256 dropout->fc10
  - model tự tạo:  conv3-32 ->maxpool -> conv3-64->maxpool -> flatten dropout-> fc512 dropout  ->fc10
  
