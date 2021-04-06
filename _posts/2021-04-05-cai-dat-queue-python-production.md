---
title: Triển khai hàng đợi xử lý bằng python với Redis
author: Quy Nguyen
date: 2021-04-05 10:09:00 +0700
categories: [Lập Trình]
tags: [queue, redis]
---

# Hàng đợi queue là gì?

- Các hệ thống chuyên ghi với cường độ lớn (như log, tracking,…). Đẩy vào xử lý queue và background job sẽ giảm nguy cơ quá tải cho database.

- Các hệ thống chuyên đọc nhưng có tính chất báo cáo, report, số lượng request ít nhưng rất mất thời gian tổng hợp.

- Các hệ thống có thời gian phản hồi lâu vì tính chất công việc, giới hạn khách quan,… Việc phản hồi cho user ngay tức thì rồi chạy trong nền sẽ tạo trải nghiệm người dùng tốt hơn. Hệ thống cũng có khả năng phục vụ nhiều user hơn.

- Các công việc phát sinh từ nghiệp vụ chính, làm việc với nhiều service ngoài nhưng không critical. Ví dụ thu thập lịch sử hệ thống, gửi email, cập nhật thông tin từ các nguồn,…

- Các công việc mang tính độc lập và ít ảnh hưởng bởi dây chuyền hay thứ tự. Đảm bảo điều này để có thể scale hệ thống bằng cách thêm nhiều worker cùng lúc.

# Vì sao phải cần đến Queue

- Hệ thống của bạn xử lý 1 công việc mất 0.5 giây, không có gì phải bàn.

- Hệ thống của bạn xử lý 1 công việc mất 10 giây. Và người dùng sẽ phải ngồi nhìn trình duyệt quay quay trong 10s để biết có gì xảy ra tiếp theo.

- Hệ thống cùng lúc chỉ có thể mở 100 kết nối. Vậy kết nối thứ 101 sẽ phải chờ 10s,… rồi kết nối phía sau sẽ bị chờ đợi, chờ đợi hoài, chờ đợi mãi,… rồi timeout.

- Và đó là lúc bạn phải nghĩ tới queue và background job. Bạn chỉ cần mất 0.5s để ghi lại yêu cầu của khách hàng vào queue, phản hồi lại họ rằng bạn sẽ xử lý, rồi ngắt kết nối với họ và tạo các background job xử lý yêu cầu này trên các worker.

# Các khái niệm
## Job
Job là các công việc cần xử lý, ví dụ như việc gửi email, ghi log… được thực hiện dưới nền dưới sự điều hành của worker
## Worker
Worker là những thành phần riêng biệt và thường là các process hoặc service xử lý một số công việc chuyên biệt nào đó. Có thể có nhiều worker cùng làm việc để xử lý các công việc

# Triển khai bằng python

Mình sử dụng RQ (Redis Queue)  để xử lý các tác vụ chạy trong nền bằng python. Thư viện này khá dễ sử dụng, và ưu điểm là nó dùng redis để đưa các job vào hàng đợi

## Cài đặt thư viện

```bash
pip install rq
```

RQ chạy trên redis nên bạn cần cài Redis trước. Hướng dẫn cài redis [tại đây](https://redis.io/topics/quickstart)

## Tạo worker

```python
import os import redis
from rq import Worker, Queue, Connection
listen = ['high', 'default', 'low']
redis_url = os.getenv('REDISTOGO_URL', 'redis://localhost:6379')
conn = redis.from_url(redis_url)
if __name__ == '__main__':
    with Connection(conn):
        worker = Worker(map(Queue, listen))
        worker.work()
```

Giờ chúng ta chạy thử worker bằng lệnh

```bash
python worker.py
```

## Viết hàm để xử lý các job

```python
import requests
def count_words_at_url(url):
    resp = requests.get(url)
    return len(resp.text.split())
```

Function này đơn giản chỉ là thực hiện request đến 1 URL sau đó sẽ đếm chiều dài của chuỗi trả về trong kết quả. Hàm này sẽ là hàm xử lý mỗi khi job được đẩy vào queue.

## Tạo hàng đợi

```python
from rq import Queue
from worker import conn
q = Queue(connection=conn)
```

Chỗ này sẽ tạo ra hàng đợi q với tham số truyền vào là đối kết nối của redis

## Đẩy job vào cho hàng đợi

```python
from utils import count_words_at_url
	result = q.enqueue(count_words_at_url, 'http://codecamp.vn)
```

Sau khi được enqueue, công việc sẽ được đẩy vào queue và worker sẽ thực hiện. Bạn theo dõi các worker sẽ thấy các tiến trình đang hoạt động.

Sau khi thực worker thực hiện xong các job, nó sẽ tiếp tục lắng nghe hàng đợi và xử lý khi có job tiếp theo. Trong 1 thời điểm, mỗi worker chỉ xử lý duy nhất 1 job và xử lý theo cơ chế tuần tự. Vì vậy giả sử mỗi job xử lý mất 5s thì nếu có 10 job, queue sẽ mất 50s để xử lý. Vậy có cách nào để tăng số lượng worker xử lý lên không. Câu trả lời là Có.

# Tăng số lượng worker để xử lý

```python
def start_worker():
	Worker([q], connection=conn).work()
```

Hàm này sẽ tạo ra 1 worker để xử lý, ý tưởng bây giờ là chúng ta sẽ tạo thêm nhiều process và mỗi process tương ứng với 1 worker.

```python
import multiprocessingfrom multiprocessing
import Process
NUM_WORKERS = 10
procsfor i in range(NUM_WORKERS):
	proc = Process(target=start_worker)
    procs.append(proc)
    proc.start()
```

Ở đây mình sử dụng 10 process tương ứng với 10 worker cùng làm việc.

Nếu quá trình chạy có lỗi hệ điều hành không cho phép python chạy multiprocess, bạn chạy thêm lệnh này để cho phép

> export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# Khi nào thì không nên dùng?

Còn sau đây là những ví dụ không khuyến khích sử dụng queue và background job trừ khi có kiến trúc phù hợp:

- Các hệ thống chuyên đọc có tính chất hoạt động như các hệ thống đọc bài viết, sản phẩm,… Các hệ thống này sẽ được optimize bằng con đường khác.
- Các công việc mang tính chất quan trọng, có tính quyết định nhưng có thời gian phản hồi không quá dài. Ví dụ các request liên quan tới việc thanh toán, tranh giành unique key hệ thống (như đặt chỗ, mua sản phẩm)

Một vài ví dụ thực tế

Đây là một số hệ thống thực tế mình đã áp dụng queue và background job. Các bạn có thể tham khảo để có cái nhìn sát hơn:

- Hệ thống tracking: client gọi lên tracking system để ghi lại các thông tin người dùng như thông tin trình duyệt, ngày giờ truy cập, ip,… Khi đó hệ thống API sẽ đẩy thông tin đó vào queue trên redis trước, sau đó response OK cho client. Sau đó worker mới lấy thông tin từ queue và ghi vào database. Do số lượng request rất lớn và lượng data ghi vào db rất nhiều nên dùng queue sẽ tránh quá tải database
- Hệ thống logging: Sau hoạt động gì đó của user với hệ thống API, như login, logout, change profile, update post, view post,… hệ thống sẽ phát sinh ra các event, nhưng các event này không được xử lý ngay mà đưa vào queue nhằm tránh block hoạt động của người dùng.
- Hệ thống notification: Hệ thống này phụ trách việc gửi thông báo, sms, email cho người dùng. Đây là công việc phát sinh từ nghiệp vụ chính, không criticalvà sử dụng nhiều service bên ngoài nên sẽ được đẩy vào queue và gửi lần lượt.
- Hệ thống analytic: đây là hệ thống báo cáo nội bộ, tuy số lượng báo cáo không nhiều, nhưng mỗi báo cáo lại rất tốn thời gian để phản hồi vì phải tổng hợp số liệu. Do vậy đẩy vào queue để tạo trải nghiệm người dùng tốt hơn. Người dùng sẽ gửi yêu cầu xuất báo cáo và nhận được phản hồi đang xử lý. Sau khi được các worker xử lý xong sẽ thông báo lại cho user qua notification.

# Nguồn tham khảo:

- Tài liệu từ RQ https://python-rq.org/docs/
- Tham khảo từ Viblo [Link](https://techtalk.vn/background-job-va-queue-cho-nguoi-nong-dan.html)
