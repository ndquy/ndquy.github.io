---
title: Huấn luyện mô hình Linear Regression
author: Quy Nguyen
date: 2021-01-04 04:45:00 +0700
categories: [Machine Learning]
tags: [Machine learning]
math: true
---

Trong các bài trước chúng ta đã xử lý một số bài toán machine learning bằng 1 số thuật toán, tuy nhiên chúng ta mới chỉ coi chúng như các hộp đen. Thậm chí chúng ta không biết gì về cách thực hiện của các thuật toán đó nhưng vẫn có thể  áp dụng, thậm chí tối ưu và cải thiện.

Tuy nhiên nếu biết được chính xác những gì thuật toán làm việc sẽ giúp chúng ta tìm được ra mô hình phù hợp với các bộ siêu tham số đủ tốt. Biết được bản chất sẽ giúp chúng ta xác định được vấn đề gặp phải và phân tích lỗi hiệu quả hơn. Trong phần này mình sẽ đi sâu vào thuật toán hồi quy tuyến tính (Linear Regression), một trong những thuật toán đơn giản nhất. Chúng ta sẽ nghiên cứu 2 phương pháp để huấn luyện:

* Sử dụng công thức để tìm ra các tham số cho mô hình mà nó khớp với dữ liệu huấn luyện nhất (nghĩa là các tham số mô hình sẽ tối thiểu hóa hàm chi phí (cost function) thông qua tập huấn luyện)
* Sử dụng phương pháp tối ưu lặp đi lặp lại, phương pháp này được gọi là Gradient Descent (xin phép không dịch), nó sẽ tối thiểu hóa dần dần hàm chi phí  thông qua quá trình huấn luyện, sau 1 số bước lặp hữu hạn, mô hình sẽ hội tụ và ta được bộ tham số giống với phương pháp đầu tiên. Chúng ta sẽ nghiên cứu 1 số thuật toán Gradient Descent: Batch GD, Mini-batch GD và Stochastic GD.

#Mô hình hồi quy tuyến tính (Linear Regression)

Mục tiêu của giải thuật hồi quy tuyến tính là dự đoán giá trị của một hoặc nhiều biến mục tiêu liên tục (continuous target variable) y dựa trên một véc-tơ đầu vào X

## Định nghĩa mô hình

$$\hat y = \theta_0 + \theta_1x_1 + \theta_2x_2 + ... + \theta_nx_n $$

