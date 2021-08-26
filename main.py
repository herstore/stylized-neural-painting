from flask import Flask, request, jsonify
import base64
import os
import argparse
import torch.optim as optim
from painter import *
import argparse
import torch
import torch.optim as optim

from painter import *

app = Flask(__name__)
# 创建一个与当前工作目录平级的文件夹用于存放照片
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'upload')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

parser = argparse.ArgumentParser(description='STYLIZED NEURAL PAINTING')
parser.add_argument('--img_path', type=str, default='./upload/uploaded_image.jpg', metavar='str')
parser.add_argument('--renderer', type=str, default='oilpaintbrush', metavar='str')
parser.add_argument('--canvas_color', type=str, default='black', metavar='str')
parser.add_argument('--canvas_size', type=int, default=512, metavar='str')
parser.add_argument('--keep_aspect_ratio', action='store_true', default=False)
parser.add_argument('--max_m_strokes', type=int, default=500, metavar='str')
parser.add_argument('--m_grid', type=int, default=5, metavar='N')
parser.add_argument('--beta_L1', type=float, default=1.0)
parser.add_argument('--with_ot_loss', action='store_true', default=False)
parser.add_argument('--beta_ot', type=float, default=0.1,
                    help='weight for optimal transportation loss (default: 0.1)')
parser.add_argument('--net_G', type=str, default='zou-fusion-net-light', metavar='str')
parser.add_argument('--renderer_checkpoint_dir', type=str, default=r'./checkpoints_G_oilpaintbrush_light',
                    metavar='str')
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--output_dir', type=str, default=r'./output', metavar='str')
parser.add_argument('--disable_preview', action='store_true', default=False)
args = parser.parse_args()
# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def optimize_x(pt):

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

    v = pt.x.detach().cpu().numpy()
    # pt._save_stroke_params(v)
    v_n = pt._normalize_strokes(pt.x)
    v_n = pt._shuffle_strokes_and_reshape(v_n)
    final_rendered_image = pt._render(v_n, save_jpgs=False, save_video=False)


@app.route('/upload_image', methods=['POST'])
def upload_image():
    json_data = request.json  # 获取上传的 JSON 数据
    if 'image_base64' in json_data:
        image_base64 = json_data['image_base64']  # 获取图片的 base64 编码
        # 将 base64 编码转换为图片文件
        image_data = base64.b64decode(image_base64)
        # 生成保存图片的文件名（假设图片格式为 jpg）
        filename = 'uploaded_image.jpg'
        # 保存图片到指定目录
        image_path = os.path.join(UPLOAD_FOLDER, filename)


        with open(image_path, 'wb') as f:
            f.write(image_data)
            # 将处理后的图片转换为 Base64 编码并返回给客户端

        pt = Painter(args=args)
        optimize_x(pt)

        #
        # # 处理上传的图片并进行模型预测
        # image = preprocess_image(image_path)
        # result = predict_image(image)

        # 处理完成后，可以返回相应的结果给客户端
        return jsonify({
            "success": True,
            "message": " Image uploaded and processed successfully.",
            "filename": filename,
            # "result": result,  # 返回模型预测结果
        })
    app.config['PROPAGATE_EXCEPTIONS'] = True
    success = True  # 表示处理是否成功，你可以根据实际处理结果设置该值
    message = "Image uploaded successfully."
    response_data = {
        "success": success,
        "message": message,

    }
    return jsonify(response_data)

@app.route('/get_image', methods=['GET'])
def get_image():
    # 假设图片文件名为 uploaded_image.jpg
    image_path = os.path.join('output', 'uploaded_image_final.png')
    if not os.path.isfile(image_path):
        response_data = {
            "success": False,
            "message": "Image not found.",
        }
        return jsonify(response_data), 404

    # 读取图片文件并将其转换为 Base64 编码
    with open(image_path, 'rb') as f:
        image_data = f.read()
    image_base64 = base64.b64encode(image_data).decode('utf-8')

    # 返回图片的 Base64 编码给客户端
    response_data = {
        "success": True,
        "image_base64": image_base64,
    }
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=False)
