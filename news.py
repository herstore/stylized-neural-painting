from flask import Flask, request, jsonify
import base64
import os
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
import argparse
import torch.optim as optim

from demo import optimize_x
from painter import *
from painter import Painter  # 导入 Painter 类
app = Flask(__name__)

# 创建一个与当前工作目录平级的文件夹用于存放照片
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'upload_images')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# setting
parser = argparse.ArgumentParser(description='demo')
parser.add_argument('--img_path', type=str, default='./upload_images/uploaded_image.jpg', metavar='str',help='path to test image (default: ./upload_images/uploaded_image.jpg)')
parser.add_argument('--renderer', type=str, default='oilpaintbrush', metavar='str', help='renderer: [watercolor, markerpen, oilpaintbrush, rectangle (default oilpaintbrush)')
parser.add_argument('--canvas_color', type=str, default='black', metavar='str', help='canvas_color: [black, white] (default black)')
parser.add_argument('--canvas_size', type=int, default=512, metavar='str', help='size of the canvas for stroke rendering')
parser.add_argument('--keep_aspect_ratio', action='store_true', default=False, help='keep input aspect ratio when saving outputs')
parser.add_argument('--max_m_strokes', type=int, default=500, metavar='str', help='max number of strokes (default 500)')
parser.add_argument('--m_grid', type=int, default=5, metavar='N', help='divide an image to m_grid x m_grid patches (default 5)')
parser.add_argument('--beta_L1', type=float, default=1.0, help='weight for L1 loss (default: 1.0)')
parser.add_argument('--with_ot_loss', action='store_true', default=False, help='imporve the convergence by using optimal transportation loss')
parser.add_argument('--beta_ot', type=float, default=0.1, help='weight for optimal transportation loss (default: 0.1)')
parser.add_argument('--net_G', type=str, default='zou-fusion-net-light', metavar='str', help='net_G: plain-dcgan, plain-unet, huang-net, zou-fusion-net, ' 
                    'or zou-fusion-net-light (default: zou-fusion-net-light)')
parser.add_argument('--renderer_checkpoint_dir', type=str, default=r'./checkpoints_G_oilpaintbrush_light', metavar='str',
                    help='dir to load neu-renderer (default: ./checkpoints_G_oilpaintbrush_light)')
parser.add_argument('--lr', type=float, default=0.002,
                    help='learning rate for stroke searching (default: 0.005)')
parser.add_argument('--output_dir', type=str, default=r'./output', metavar='str', help='dir to save painting results (default: ./output)')
parser.add_argument('--disable_preview', action='store_true', default=False, help='disable cv2.imshow, for running remotely without x-display')
args = parser.parse_args()

# 调用函数
def optimize_x(pt):
    # Decide which device we want to run on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pt._load_checkpoint()
    pt.net_G.eval()

    pt.initialize_params()
    pt.x_ctt.requires_grad = True
    pt.x_color.requires_grad = True
    pt.x_alpha.requires_grad = True
    utils.set_requires_grad(pt.net_G, False)

    pt.optimizer_x = optim.RMSprop([pt.x_ctt, pt.x_color, pt.x_alpha], lr=pt.lr)

    print('begin to draw...')
    pt.step_id = 0
    for pt.anchor_id in range(0, pt.m_strokes_per_block):
        pt.stroke_sampler(pt.anchor_id)
        iters_per_stroke = int(500 / pt.m_strokes_per_block)
        for i in range(iters_per_stroke):

            pt.optimizer_x.zero_grad()

            pt.x_ctt.data = torch.clamp(pt.x_ctt.data, 0.1, 1 - 0.1)
            pt.x_color.data = torch.clamp(pt.x_color.data, 0, 1)
            pt.x_alpha.data = torch.clamp(pt.x_alpha.data, 0, 1)

            if args.canvas_color == 'white':
                pt.G_pred_canvas = torch.ones(
                    [args.m_grid ** 2, 3, pt.net_G.out_size, pt.net_G.out_size]).to(device)
            else:
                pt.G_pred_canvas = torch.zeros(
                    [args.m_grid ** 2, 3, pt.net_G.out_size, pt.net_G.out_size]).to(device)

            pt._forward_pass()
            pt._drawing_step_states()
            pt._backward_x()
            pt.optimizer_x.step()

            pt.x_ctt.data = torch.clamp(pt.x_ctt.data, 0.1, 1 - 0.1)
            pt.x_color.data = torch.clamp(pt.x_color.data, 0, 1)
            pt.x_alpha.data = torch.clamp(pt.x_alpha.data, 0, 1)

            pt.step_id += 1

        # 最后的渲染和保存步骤
    v = pt.x.detach().cpu().numpy()
    pt._save_stroke_params(v)
    v_n = pt._normalize_strokes(pt.x)
    v_n = pt._shuffle_strokes_and_reshape(v_n)
    final_rendered_image = pt._render(v_n, save_jpgs=True, save_video=False)
    # final_rendered_image = pt.process_image(image_data)

    # 返回渲染后的图像给客户端
    response = final_rendered_image.tobytes()  # 将渲染后的图像转换成字节数据
    return response, 200, {'Content-Type': 'image/jpeg'}  # 返回图像数据和 HTTP 状态码 200


# 预处理
def preprocess_image(image_data):
    # 将 base64 编码的图片数据解码
    image_bytes = io.BytesIO(image_data)
    image = Image.open(image_bytes)

    # 进行图片尺寸调整和归一化等操作
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图片尺寸
        transforms.ToTensor(),           # 将图片转换为 torch.Tensor 格式
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
    ])
    image_tensor = transform(image).unsqueeze(0)  # 添加 batch 维度，使其成为 4D tensor

    return image_tensor

def image_to_base64(image):
    # 将图像数据转换为 Base64 编码的字符串
    image_pil = Image.fromarray(image)  # 假设 final_rendered_image 是 numpy 数组
    image_buffer = io.BytesIO()
    image_pil.save(image_buffer, format='PNG')  # 可以根据实际格式保存为 JPEG 等
    image_base64 = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
    return image_base64

# process_image 函数中获取 optimize_x 的结果并返回
def process_image(image_data):
    # 预处理图片数据
    image_tensor = preprocess_image(image_data)
    # 创建 Painter 实例
    pt = Painter(args=args)
    # 调用 optimize_x 函数进行处理
    final_rendered_image = optimize_x(pt)
    rendered_image_base64 = image_to_base64(final_rendered_image)
    return rendered_image_base64

@app.route('/upload_image', methods=['POST'])
def upload_image():
    json_data = request.json  # 获取上传的 JSON 数据
    if 'image_base64' in json_data:
        image_base64 = json_data['image_base64']  # 获取图片的 base64 编码
        # 将 base64 编码转换为图片文件
        image_data = base64.b64decode(image_base64)

        # 调用处理函数对图片进行处理
        process_image(image_data)

        # 生成保存图片的文件名（假设图片格式为 jpg）
        filename = 'uploaded_image.jpg'
        # 保存图片到指定目录
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        with open(image_path, 'wb') as f:
            f.write(image_data)

        # 处理完成后，可以返回相应的结果给客户端
        return jsonify({
            "success": True,
            "message": "Image uploaded and processed successfully.",
            "filename": filename,
        })

    success = True  # 表示处理是否成功，你可以根据实际处理结果设置该值
    message = "Image uploaded successfully."
    response_data = {
        "success": success,
        "message": message,
    }
    return jsonify(response_data)

if __name__ == '__main__':
     app.run(debug=True)
