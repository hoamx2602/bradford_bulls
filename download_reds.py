import gdown
url = 'https://drive.google.com/uc?id=14D4V4raNYIOhETfcuuLI3bGLB-OYIv6X'
output = 'weights/NAFNet-REDS-width64.pth'
gdown.download(url, output, quiet=False)
