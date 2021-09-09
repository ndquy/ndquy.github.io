---  
title: \[Tản mạn\] Đánh giá độ chính xác các bộ kit xét nghiệm thế nào?  
author: Quy Nguyen  
date: 2021-09-09 10:48:00 +0700  
categories: [Machine Learning]  
tags: [Machine learning]  
math: true
---

Hôm trước có chủ tịch tên là NTQ của một tập đoàn công nghệ nào đấy là BK** (Mình lhoong tiện nói tên) phát triển công nghệ giúp tìm ra người nhiễm Covid-19 thông qua dung dịch nước muối sinh lý bảo là: "Kết quả ban đầu được ghi nhận là khả quan với tỷ lệ nhận diện trên 90%". Mà không nói rõ tỷ lệ này là tỷ lệ gì. Hôm nay mọi người cùng tìm hiểu thử độ chính xác của các bộ kit xét nghiệm được xác định thế nào?

# Xét nghiệm covid 19
Giả sử trên một tập người cần xét nghiệm để phát hiện có mắc covid hay không, xác suất mắc covid trên tập người đó rất thấp, chỉ 1% số người trong đó là mắc covid chẳng hạn. Nếu dùng bộ kit xét nghiệm tiến hành xét nghiệm trên toàn bộ tập người đó và cho rằng 100% số mẫu là âm tính. Nếu dùng độ chính xác thông thường thì bộ kit đã bỏ lọt 1% số mẫu dương tính, nghĩa là lúc này bộ kit đạt độ chính xác là 99%. Tuy nhiên bộ kit xét nghiệm lại không hề có tác dụng gì trong việc tìm ra người mắc covid. (trường hợp này chắc giống với ông nghệ của BK**).
# Test nhanh
Như vậy là ở đây cần phải có một phương pháp khác để đánh giá độ chính xác của một bộ kit xét nghiệm.

Chúng ta khi tạo ra một bộ kit xét nghiệm thì đều mong muốn tìm ra mẫu dương tính chính xác. Lúc này kit xét nghiệm thà xác định nhầm một mẫu từ âm tính thành dương tính còn tốt hơn là một mẫu dương tính nhầm thành âm tính. Như vậy độ chính xác của kit xét nghiệm sẽ quan tâm đến mức độ xác định chính xác một mẫu dương tính trong số các mẫu dương tính hơn là số mẫu xác định đúng trên tổng số mẫu. Lấy ví dụ, giả sử trong 100.000 người có 100 người mắc covid, thì một bộ kit với độ chính xác 90% sẽ phát hiện ra 90 người mắc trong tổng số 100 người mắc. Số trường hợp phát hiện nhầm âm tính thành dương tính có thể cao nhưng với test nhanh thì hoàn toàn có thể chấp nhận được tỉ lệ này.
# Test khẳng định
Tuy nhiên lại có thêm 1 vấn đề nữa ở đây, nếu test nhanh chấp nhận tỉ lệ âm tính nhầm thành dương tính (dương tính giả) cao thì xét nghiệm chắc chắn (tạm gọi là xét nghiệm PCR) lại không được phép nhầm như vậy. PCR phải xác định chắc chắn một mẫu phải là âm tính hay dương tính, việc âm tính giả hay dương tính giả cũng ảnh hưởng đến độ chính xác của PCR. Như vậy xét nghiệm PCR sẽ cần phải đánh giá độ chính xác dưới một phương pháp khác.

Để xác định được độ chính xác này, ta giả sử một máy xét nghiệm PCR xét nghiệm số mẫu dương tính đúng là dương tính là TP (true positive), và số mẫu âm tính đúng là âm tính là TN (true negative). 

Các trường hợp âm tính giả là FP (false positive), và dương tính giả là FN (false negative). 

![Đánh giá kết quả xét nghiệm](/assets/img/blog/xetnghiemcovid19.jpeg)  
_Đánh giá kết quả xét nghiệm_


Thông thường mọi người thường hay dùng các phương pháp sau để đánh giá:
- Precision: Độ chính xác này cho biết trong số những người được xét nghiệm dương tính, có bao nhiêu người trong số những người đó thực sự nhiễm virus. Được tính theo công thức precision = TP/(TP+FP) (sử dụng cho các bộ kit nhanh)
- Recall: Cho biết trong số những người bị nhiễm virus, có bao nhiêu người trong số những người đó có kết quả xét nghiệm dương tính. Được tính theo công thức Recall = TP/(TP+FN)
- Specificity: Cho biết xét nghiệm dự đoán chính xác rằng một người KHÔNG mắc coronavirus là bao nhiêu. Được tính theo công thức Specificity = TN/(TN+FP)

