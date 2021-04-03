---
title: 2. Huấn luyện mô hình Linear Regression
author: Quy Nguyen
date: 2021-04-02 04:45:00 +0700
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

Với đầu vào x, đầu ra là y. Ta có mô hình như sau:

$$\hat y = \theta_0 + \theta_1x_1 + \theta_2x_2 + ... + \theta_nx_n $$

Trong đó
* $\( \hat y \)$ là giá trị đầu ra cần dự đoán

* $n$ là số lượng features (số chiều của vector đầu vào X)

* $\( x_i \)$ là feature thứ i.

* $\( \theta \)$ là tham số của mô hình (bao gồm $\( \theta_0 \)$ là bias và các  tham số $\( \theta_1, \theta_2 \),...)$

Phương trình này có thể viết ngắn gọn lại thành:

$$\hat y = h_\theta(x) = \theta . x$$

Trong đó:

* $\( \theta  \)$ là vector tham số mô hình bao gồm bias $\( \theta_0  \)$ và các feature weights $\( \theta_1 ... \theta_n\)$

* X là vector đầu vào của mô hình, trong đó $\( x_0\)$ có giá trị bằng 1 (được thêm vào)

* $θ · x$ là tích có hướng của vector đầu vào và trọng số

Trong Machine learning, các vector thường được biểu diễn dưới dạng các vector cột, dạng mảng 2 chiều với 1 cột. Nếu $θ$ và $x$ là các vector cột thì $\( \hat y = \theta^Tx \)$  trong đó $\( \theta^T \)$ là chuyển vị của $θ$

## Huấn luyện mô hình

Sau khi đã có mô hình hồi quy tuyến tính, giờ làm cách nào để huấn luyện chúng? Mình nhắc lại 1 chút, đó là việc huấn luyện cho mô hình là việc tìm ra các tham số tối ưu nhất thông qua bộ dữ liệu huấn luyện. Để làm được điều này chúng ta phái xác định được thước đo để biết được mô hình tốt hay không tốt (tham số có tối ưu hay không). Sau đây mình sẽ giới thiệu 1 cách để xác định đó là dùng biểu thức Root Mean Square Error (RMSE). Hiểu nôm na nó là Bình phương lỗi trung bình. Và mục tiêu của việc huấn luyện đó là giảm thiểu giá trị của Mean Square Error (MSE).

MSE của mô hình hồi quy tuyến tính $\( h_\theta \)$ trên tập huấn luyện X được tính như sau:

$$ MSE(X, h_\theta) = \frac{1}{m}\sum_{i=1}^m(\theta^Tx^{(i)} -y^{(i)})^2$$

Chúng ta viết $\( h_\theta \)$ thay vì h để làm rõ việc mô hình được tham số hóa bởi vector $θ$ (phụ thuộc vào  $θ$). Để đơn giản công thức, chúng ta sẽ chỉ viết $MSE(θ)$ thay vì  $MSE(X, hθ)$

## Phương trình chuẩn

Để tìm giá trị θ để phương trình $MSE(θ)$  đạt cực tiểu có thể tìm được nhờ công thức sau:
$$ \hat\theta = (X^TX)^{-1}  X^T  y $$

Phương trình (5.2)

Trong đó:

$\(\hat\theta\)$ là giá trị của θ mà tại đó $MSE(θ)$ đạt cực tiểu
$y$ là vector giá trị cần tìm bao gồm $\( y^1...y^m \)$

# Implement bằng python

Khởi tạo dữ liệu

```python
import numpy as np
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
```

Tiếp theo chúng ta sẽ tính θ dựa vào công thức 5.2. Chúng ta sẽ sử dụng hàm inv() trong thư viện Linear Algebra  của numpy để tính nghịch đảo của ma trận và hàm dot() để nhân ma trận:

```python
X_b = np.c_[np.ones((100, 1)), X] # thêm x0 = 1 cho mỗi giá trị của vector X
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
```

Cùng xem kết quả:

```
>>> theta_best
array([[4.21509616],
[2.77011339]])
```

Chúng ta sẽ kỳ vọng là $\( \theta_0 = 4 và \theta_1 = 3 \)$ thay vì $\( \theta_0 = 4.215 và \theta_1 = 2.770 \)$. Tuy nhiên khi khởi tạo dữ liệu chúng ta đã thêm vào 1 ít noise nên không thể đưa nó về dạng phương trình gốc được.

Sau khi đã có $\( \hat\theta \)$, chúng ta có thể dự đoán kết quả với đầu vào mới:

```bash
>>> X_new = np.array([[0], [2]])
>>> X_new_b = np.c_[np.ones((2, 1)), X_new] # add x0 = 1 to each instance >>> y_predict = X_new_b.dot(theta_best)
>>> y_predict
array([[4.21509616],
[9.75532293]])
```

Giờ chúng ta sẽ vẽ đường thẳng cho các dự đoán

```python
plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()
```

Kiểm tra lại bằng thư viện trong Scikit-Learn

```bash
>>> from sklearn.linear_model
import LinearRegression
>>> lin_reg = LinearRegression()
>>> lin_reg.fit(X, y)
>>> lin_reg.intercept_, lin_reg.coef_ (array([4.21509616]), array([[2.77011339]]))
>>> lin_reg.predict(X_new) array([[4.21509616],
[9.75532293]])
```

# Tổng kết
Vậy là bài này mình đã hướng dẫn các bạn cách huấn luyện thuật toán hồi quy tuyến tính. Chốt lại về cơ bản, để huấn luyện 1 thuật toán cần:
* Định nghĩa được mô hình
* Định nghĩa hàm chi phí (hoặc hàm mất mát)
* Tối ưu hàm chi phí bằng dữ liệu huấn luyện
* Tìm ra các trọng số của mô hình mà tại đó hàm chi phí có giá trị nhỏ nhất
