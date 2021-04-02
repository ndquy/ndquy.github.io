---
title: Softmax Regression
author: Quy Nguyen
date: 2021-04-02 04:45:00 +0700
categories: [Machine Learning]
tags: [Machine learning]
math: true
---

Softmax regression (hay còn gọi là multinomial logistic regression) là dạng của hồi quy logistic cho trường hợp cần phân loại nhiều lớp. Trong hồi quy logistic chúng ta giả sử rằng các nhãn là các giá trị nhị phân $ y^{(i)} \in \{0,1\}$. Softmax regression cho phép chúng ta thực hiện phân loại $ y^{(i)} \in \{1,\ldots,K\} $ với K là số lớp cần dự đoán.

# Định nghĩa mô hình

Chúng ta có tập dữ liệu huấn luyện $\{ (x^{(1)}, y^{(1)}), \ldots, (x^{(m)}, y^{(m)}) \}$ với m dữ liệu được đánh nhãn với input features x $ x^{(i)} \in \Re^{n}$. Với hồi quy logistic chúng ta có mô hình phân loại nhị phân, vì vậy $y^{(i)} \in \{0,1\} $. Chúng ta có giả thiết sau:

$$\begin{align} h_\theta(x) = \frac{1}{1+\exp(-\theta^\top x)}, \end{align}$$
Và tham số mô hình θ đã được huấn luyện để tối ưu hàm chi phí

$$\begin{align} J(\theta) = -\left[ \sum_{i=1}^m y^{(i)} \log h_\theta(x^{(i)}) + (1-y^{(i)}) \log (1-h_\theta(x^{(i)})) \right] \end{align}$$

Trong  softmax regression, chúng ta quan tâm tới việc phân loại nhiều lớp và nhãn y có thể là 1 trong K giá trị khác nhau thay vì chỉ 2 như logistic. Vì vậy trong tập huấn luyện $\{ (x^{(1)}, y^{(1)}), \ldots, (x^{(m)}, y^{(m)}) \}$, chúng ta có $y^{(i)} \in \{1, 2, \ldots, K\}$ (chú ý chúng ta quy ước sẽ đánh chỉ số cho các lớp bắt đầu từ 1, thay vì từ 0). Ví dụ trong nhận diện số viết tay dùng tập MNIST, chúng ta sẽ có K=10 lớp khác nhau.

Cho 1 dữ liệu đầu vào x, chúng ta cần phải ước lượng được xác xuất thuộc vào 1 lớp nào đó P(y=k | x) với $k = 1, \ldots, K$. Sẽ có K giá trị xác suất khác nhau, vì vậy giả thiết của chúng ta sẽ đưa ra vector K chiều gồm các giá trị xác suất. Cụ thể, giả thiết $h_{\theta}(x)$ sẽ có dạng:

$$\begin{align} h_\theta(x) = \begin{bmatrix} P(y = 1 | x; \theta) \\ P(y = 2 | x; \theta) \\ \vdots \\ P(y = K | x; \theta) \end{bmatrix} = \frac{1}{ \sum_{j=1}^{K}{\exp(\theta^{(j)\top} x) }} \begin{bmatrix} \exp(\theta^{(1)\top} x ) \\ \exp(\theta^{(2)\top} x ) \\ \vdots \\ \exp(\theta^{(K)\top} x ) \\ \end{bmatrix} \end{align}$$

Với $ \theta^{(1)}, \theta^{(2)}, \ldots, \theta^{(K)} \in \Re^{n}$ là các tham số của mô hình. Để ý tổng $\frac{1}{ \sum_{j=1}^{K}{\exp(\theta^{(j)\top} x) } }$ ta tiến hành nhân vào để chuẩn hóa phân phối, vì vậy tổng các phần tử của $ h_\theta(x)$ sẽ bằng 1.

Để thuận tiện, chúng ta sẽ viết $\theta$ đại diện cho các tham số của mô hình. Khi thực hiện implement bằng code sẽ dễ hơn biểu diễn $\theta$ bằng ma trận nxK, $\theta$  thu được bằng cách gộp $\theta^{(1)}, \theta^{(2)}, \ldots, \theta^{(K)}$ vào các cột như sau:

$$\theta = \left[\begin{array}{cccc}| & | & | & | \\ \theta^{(1)} & \theta^{(2)} & \cdots & \theta^{(K)} \\ | & | & | & | \end{array}\right]$$
