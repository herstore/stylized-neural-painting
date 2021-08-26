# import argparse
import torch
import torch.optim as optim
import argparse
from painter import *


part1 = './upload_images/uploaded_image.jpg'
part2 = 'oilpaintbrush'
part3 = 'black'
part4 = 512
part5 = 'store_true'
part6 = 500
part7 = 5
part8 = 1.0
part9 = False
part10 = 0.1
part11 = 'zou-fusion-net-light'
part12 = r'./checkpoints_G_oilpaintbrush_light'
part13 = 0.002
part14 = r'./output'
part15 = 'store_true'
args ={}
args = part1, part2, part3, part4, part5, part6, part7, part8, part9, part10, part11, part12, part13, part14, part15

args = argparse.Namespace(**args)
print(type(args))  # 输出 <class 'int'>


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
    pt._save_stroke_params(v)
    v_n = pt._normalize_strokes(pt.x)
    v_n = pt._shuffle_strokes_and_reshape(v_n)
    final_rendered_image = pt._render(v_n, save_jpgs=True, save_video=True)



if __name__ == '__main__':
    # 将args字典中的--renderer参数作为键值对传递给Painter类的构造函数
    pt = Painter(args=args)

    optimize_x(pt)
