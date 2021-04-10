---
title: 12. Các phương pháp đánh giá mô hình phân lớp phần 1
author: Quy Nguyen
date: 2021-04-10 15:47:00 +0700
categories: [Machine Learning]
tags: [Machine learning]
---

Trong bài viết này mình sẽ nói đến bài toán phân lớp và các phương pháp đánh giá 1 mô hình phân lớp.

# Bài toán phân lớp

Mình sẽ sử dụng bộ dữ liệu MNIST, gồm 70.000 ảnh nhỏ của các số viết tay bởi người ở US. Mỗi ảnh được đánh nhãn với số tương ứng. Tập dữ liệu này được dùng cực kì phổ biến trong huấn luyện các thuật toán và thường được gọi là bộ dữ liệu "Hello World" trong Machine learning. Nói chung là ai học machine learning thì sớm hay muộn cũng phải sử dụng MNIST =))

## Dữ liệu huấn luyện

Scikit-Learn cung cấp nhiều functions để tải về các bộ dữ liệu để huấn luyện. Trong đó có MNIST. Đoạn code sau đây để tải về dataset:

```python
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
mnist.keys()
dict_keys(['data', 'target', 'feature_names', 'DESCR', 'details',
'categories', 'url'])
```

Sau đó xem kết quả

```
Let’s look at these arrays:
>>> X, y = mnist["data"], mnist["target"] >>> X.shape
(70000, 784)
>>> y.shape
(70000,)
```

Có 70k ảnh và mỗi ảnh có 784 features. Bởi vì mỗi ảnh có 28x28 pixels và mỗi feature đơn giản được biểu diễn bởi 1 màu từ 0 (white) đến 255 (black).



Thử in thử 1 ảnh trong bộ dataset để xem:

```python
import matplotlib as mpl import matplotlib.pyplot as plt
some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest") plt.axis("off")
plt.show()
```


Bây giờ ta thử xem 1 vài mẫu trong tập MNIST:

![Dataset](/assets/img/blog/Ten-samples-for-each-digit-in-the-MNIST-dataset.png)
_Dataset MNIST_

Phân chia tập dữ liệu, chúng ta sẽ tiến hành chia bộ dữ liệu ra làm 2 phần: 1 phần để training (huấn luyện) gồm 60k ảnh đầu tiên và 1 phần để đánh giá (test) gồm 10k ảnh cuối của tập dữ liệu.

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
Huấn luyện bộ phân lớp nhị phân (Binary Classifier)
Để cho đơn giản, chúng ta sẽ tiến hành phân lớp với 1 số, trong ví dụ này là số 5. Bộ phát hiện số 5 được gọi là 1 bộ phân lớp nhị phân (đúng hoặc sai)

## Chuẩn bị dữ liệu
Bây giờ chúng ta sẽ tạo tập dữ liệu để huấn luyện:
```python
#  y được gán nhãn là True nếu nhãn của y là số 5, False nếu nhãn không phải số 5
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
```
## Xây dựng và huấn luyện mô hình
Sau khi đã có tập dữ liệu để huấn luyện, bây giờ chúng ta sẽ xác định bộ phân lớp phù hợp để thực hiện phân loại. Ở bài viết này mình sử dụng bộ phân lớp Stochastic Gradient Descent (SGD)

```python
from sklearn.linear_model import SGDClassifier sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
```

SGDClassifier dựa vào việc lấy ngẫu nhiên trong quá trình training (do đó được stochastic). Nếu bạn muốn kết quả không đổi sau mỗi lần chạy, bạn nên đặt thêm tham số random_state
Dự đoán kết quả sau khi huấn luyện
Sau khi huấn luyện xong chúng ta sẽ thực hiện chạy thử mô hình.

```python
sgd_clf.predict([some_digit])
# array([ True])
```

Sau khi đã chạy xong việc huấn luyện mô hình, chúng ta sẽ đi vào đánh giá độ chính xác mô hình trong việc dự đoán.

# Các phương pháp đánh giá mô hình dự đoán

## Cross-validation.

Phương pháp tốt nhất để đánh giá 1 mô hình học máy đó là cross-validation. Cross-validation là một phương pháp kiểm tra độ chính xác của 1 máy học dựa trên một tập dữ liệu học cho trước. Thay vì chỉ dùng một phần dữ liệu làm tập dữ liệu học thì cross-validation dùng toàn bộ dữ liệu để dạy cho máy. Ở bài này mình sẽ sử dụng K-fold, đây là phương pháp dùng toàn bộ dữ liệu và chia thành K tập con. Quá trình học của máy có K lần. Trong mỗi lần, một tập con được dùng để kiểm tra và K-1 tập còn lại dùng để dạy.

```python
from sklearn.model_selection import StratifiedKFold from sklearn.base import clone
skfolds = StratifiedKFold(n_splits=3, random_state=42)
for train_index, test_index in skfolds.split(X_train, y_train_5):
  clone_clf = clone(sgd_clf)
  X_train_folds = X_train[train_index]
  y_train_folds = y_train_5[train_index]
  X_test_fold = X_train[test_index] y_test_fold = y_train_5[test_index]
  clone_clf.fit(X_train_folds, y_train_folds)
  y_pred = clone_clf.predict(X_test_fold)
  n_correct = sum(y_pred == y_test_fold)
  print(n_correct / len(y_pred)) # Lần lượt là 0.9502, 0.96565 và 0.96495
```

Để rút gọn thì thư viện sklearn đã cung cấp sẵn hàm để thực hiện:
```python
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
# array([0.96355, 0.93795, 0.95615])
```


## Confusion Matrix

Một phương pháp tốt hơn để đánh giá performance của mô hình phân lớp đó là confusion matrix (ma trận nhầm lẫn). Ý tưởng chính là đếm số lần phần tử thuộc class A bị phân loại nhầm vào class B.

Để thực hiện tính toán ma trận nhầm lẫn, đầu tiên bạn phải có kết quả các dự đoán và so sánh với nhãn thật của nó. Nghĩa là chúng ta phải dự đoán trên tập test, sau đó dúng kết quả dự đoán này để so sánh với nhãn ban đầu.

```python
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
```

Sau đó xác định ma trận nhầm lẫn:

```python
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)
# array([[53057, 1522],
#		[ 1325,  4096]])
```

Ma trận nhầm lẫn sẽ cho chúng ta nhiều thông tin về chất lượng của bộ phân lớp.

* TP (True Positive): Số lượng dự đoán chính xác. Là khi mô hình dự đoán đúng một số là số 5.
* TN (True Negative): Số lương dự đoán chính xác một cách gián tiếp. Là khi mô hình dự đoán đúng một số không phải số 5, tức là việc không chọn trường hợp số 5 là chính xác.
* FP (False Positive - Type 1 Error): Số lượng các dự đoán sai lệch. Là khi mô hình dự đoán một số là số 5 và số đó lại không phải là số 5
* FN (False Negative - Type 2 Error): Số lượng các dự đoán sai lệch một cách gián tiếp. Là khi mô hình dự đoán một số không phải số 5 nhưng số đó lại là số 5, tức là việc không chọn trường hợp số 5 là sai.


Từ 4 chỉ số này, ta có 2 con số để đánh giá mức độ tin cậy của một mô hình:

### Precision and Recall

* Precision: Trong tất cả các dự đoán Positive được đưa ra, bao nhiêu dự đoán là chính xác? Chỉ số này được tính theo công thức

```
precision = TP  / (TP + FP)
```
* Recall: Trong tất cả các trường hợp Positive, bao nhiêu trường hợp đã được dự đoán chính xác? Chỉ số này được tính theo công thức:

```
recall = TP  / (TP + FN)
```

```python
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5, y_train_pred)
# == 4096 / (4096 + 1522) 0.7290850836596654
recall_score(y_train_5, y_train_pred)
# == 4096 / (4096 + 1325) 0.7555801512636044
```

### F1-SCORE

Để kết hợp 2 chỉ số này, người ta đưa ra chỉ số F1-score

Một mô hình có chỉ số F-score cao chỉ khi cả 2 chỉ số Precision và Recall để cao. Một trong 2 chỉ số này thấp đều sẽ kéo điểm F-score xuống. Trường hợp xấu nhất khi 1 trong hai chỉ số Precison và Recall bằng 0 sẽ kéo điểm F-score về 0. Trường hợp tốt nhất khi cả điểm chỉ số đều đạt giá trị bằng 1, khi đó điểm F-score sẽ là 1.

Để tính F1-score, ta thực hiện như sau:

```python
from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)
# 0.7420962043663375
```

Tuy nhiên thì không phải lúc nào ta cũng cần đến F1, 1 vài trường hợp ta chỉ quan tâm đến precision, 1 vài trường hợp ta quan tâm đến recall. Ví dụ, nếu bạn huấn luyện 1 mô hình để phát hiện video an toàn cho trẻ em, bạn phải sử dụng bộ phân lớp mà có thể bỏ sót nhiều video an toàn (recall thấp) nhưng ít bỏ qua các video không an toàn (high precision). Hay còn gọi là giết nhầm còn hơn bỏ sót, thà không hiển thị video an toàn còn hơn là hiển thị video không an toàn.

# Source Code

Các bạn có thể xem tại: [Github](https://github.com/dinhquy94/codecamp.vn/blob/master/bai3_4.ipynb)

# Nguồn tham khảo:

* [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)

