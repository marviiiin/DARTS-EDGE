# coding: utf-8
from http.server import HTTPServer, BaseHTTPRequestHandler
from multiprocessing import Process
import cgi
import portalocker

def handle_post(request_handler):
    try:
        content_length = int(request_handler.headers['Content-Length'])
        if content_length > 1e6:  # 限制请求体的大小，防止DoS攻击
            raise ValueError('Request body is too large')

        ctype, pdict = cgi.parse_header(request_handler.headers.get('content-type'))
        if ctype != 'text/plain':  # 只接受纯文本格式的数据
            request_handler.send_error(400, 'Bad request: only text/plain data is allowed')
            return

        post_data = request_handler.rfile.read(content_length)
        post_data_str = post_data.decode('utf-8')

        with open('requests.txt', 'a') as f:
            portalocker.lock(f, portalocker.LOCK_EX)  # 使用portalocker来锁定文件
            f.write(post_data_str + '\n')
            portalocker.unlock(f)  # 解锁文件

        request_handler.send_response(200)
        request_handler.end_headers()
        request_handler.wfile.write(b'Post received!')

    except Exception as e:
        request_handler.send_error(500, 'Internal server error: ' + str(e))

def request_handler_factory(stop_event):
    class RequestHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if stop_event.is_set():
                self.send_error(503, 'Server is shutting down')
            else:
                handle_post(self)

    return RequestHandler

def gpsFetchProcess(stop_event):
    server = HTTPServer(('localhost', 8000), request_handler_factory(stop_event))
    while not stop_event.is_set():
        server.handle_request()

