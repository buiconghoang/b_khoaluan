Dữ liệu:
  - có 46 class,  mỗi class có khoảng 1400 ảnh
  - generate (file generate data): mỗi ảnh sẽ được tạo thành 12 ảnh bao gồm: 
    + 2 ảnh xoay: random từ -45 đến 45
    + zoom:  (ảnh gốc + 2 ảnh xoay) * 3 = 9 ảnh; ảnh zoom random từ 0.75 đến 1 và zoom theo trung tâm
    + tổng = ảnh gốc + 2 ảnh xoay + 9 ảnh zoom = 12 ảnh
  - Dữ liệu được nén vào file có tên train_data.file (12gb)
  - link: https://drive.google.com/file/d/1caLpZNQvhw-dYbu3kez_a6Mi4HsGv3nn/view?usp=sharing
Model: (file train)
  - model theo bài báo: M6-1: conv3-32 -> conv3-32 ->maxpool -> conv3-64 -> conv3-64 -> maxpool ->flatten -> fc-256 dropout->fc10
  - model tự tạo:  conv3-32 ->maxpool -> conv3-64->maxpool -> flatten dropout-> fc512 dropout  ->fc10
  Mỗi model sẽ chạy 3 epochs, với batch_size là 100, tương đương với gần 17.000 vòng lặp
  
  link model M6-1: https://drive.google.com/drive/folders/135g7a61wxL3U8hv2U0iZ0mC5AX-_IbKD?usp=sharing
  link model tự tạo: https://drive.google.com/drive/folders/1Z56HHufvyPzPTTVteGAaSIuKedQPuAK4?usp=sharing
  
  model tự tạo có vẻ kết quả dự đoán chính xác hơn một xíu do tỉ lệ train và test chưa đến ngưỡng 100% còn model M6-1 thì dính ngưỡng 100% vài lần nên chắc ăn theo dữ liệu ban đầu nhiều quá.
Test: 
Xử lý dữ liệu đầu vào:
    + crop ảnh vào thành 40x40, cái này chỉ bao quanh chữ, nếu không chữ sẽ to toàn ảnh và không khớp với dữ liệu train. Dữ liệu train thì cái ảnh rơi vào khoảng 40x40, hầu như cái nào cũng có padding 
    + thêm padding 12x12 vào mỗi ảnh mình crop
Test ảnh:
    + file Test2.py và test2 jupyter notebook: test ảnh thông thường, mình đưa đường dẫn ảnh vào và chạy, cần phải chỉnh đường dẫn model, file Test2.py để import vào file giao diện
    + file app : vẽ và crop ảnh truyền vào Test2.py để dự đoán kết quả
