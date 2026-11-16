"""Render source-audited M4 diagrams as editable SVG and PNG; no model runs."""
from pathlib import Path
from html import escape
import math
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
FONT = '/System/Library/Fonts/STHeiti Light.ttc'
BOLD = '/System/Library/Fonts/STHeiti Medium.ttc'
INK, MUTED = '#182b46', '#51647a'
BLUE, TEAL, PURPLE, AMBER = '#2366ad', '#128578', '#7552b2', '#b27717'


class Canvas:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.im = Image.new('RGB', (width, height), '#f8fafc')
        self.d = ImageDraw.Draw(self.im)
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                    '<rect width="100%" height="100%" fill="#f8fafc"/>']

    def rect(self, box, fill, stroke='#d5dfeb', radius=18):
        x, y, w, h = box
        self.d.rounded_rectangle((x, y, x+w, y+h), radius, fill=fill, outline=stroke, width=2)
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')

    def text(self, x, y, text, size=26, color=INK, bold=False, anchor='lt'):
        font = ImageFont.truetype(BOLD if bold else FONT, size)
        self.d.text((x, y), text, fill=color, font=font, anchor=anchor)
        svg_anchor = 'middle' if anchor.startswith('m') else 'start'
        self.svg.append(f'<text x="{x}" y="{y}" text-anchor="{svg_anchor}" dominant-baseline="text-before-edge" font-family="STHeiti,PingFang SC,Noto Sans CJK SC,sans-serif" font-size="{size}" font-weight="{600 if bold else 400}" fill="{color}">{escape(text)}</text>')

    def box(self, box, title, lines=(), color=BLUE, fill='#eef5fc', size=25):
        self.rect(box, fill, color)
        x, y, w, h = box
        title_size = size + 3
        font = ImageFont.truetype(BOLD, title_size)
        assert self.d.textlength(title, font=font) < w-24, title
        self.text(x+w/2, y+16, title, title_size, color, True, 'mt')
        yy = y+58
        for line in lines:
            assert self.d.textlength(line, font=ImageFont.truetype(FONT, size)) < w-24, line
            self.text(x+w/2, yy, line, size, INK, anchor='mt')
            yy += size+12
        assert yy <= y+h+8, (title, yy, y+h)

    def arrow(self, points, color=MUTED, dashed=False, width=3, head=True):
        if dashed:
            for a, b in zip(points, points[1:]):
                length = math.dist(a,b)
                for i in range(0, int(length), 18):
                    s, t = i/max(length,1), min(i+10,length)/max(length,1)
                    self.d.line((a[0]+(b[0]-a[0])*s,a[1]+(b[1]-a[1])*s,a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t), fill=color, width=width)
        else:
            self.d.line(points, fill=color, width=width)
        ps = ' '.join(f'{x},{y}' for x,y in points)
        self.svg.append(f'<polyline points="{ps}" fill="none" stroke="{color}" stroke-width="{width}"'+(' stroke-dasharray="10 8"' if dashed else '')+'/>')
        if head:
            a,b = points[-2:]
            angle = math.atan2(b[1]-a[1],b[0]-a[0])
            tri = [b,(b[0]-13*math.cos(angle-.48), b[1]-13*math.sin(angle-.48)),(b[0]-13*math.cos(angle+.48),b[1]-13*math.sin(angle+.48))]
            self.d.polygon(tri, fill=color)
            self.svg.append(f'<polygon points="'+ ' '.join(f'{x},{y}' for x,y in tri)+f'" fill="{color}"/>')

    def save(self, name):
        self.im.save(OUT/f'{name}.png')
        (OUT/f'{name}.svg').write_text('\n'.join(self.svg+['</svg>']))


def overall():
    c=Canvas(1800,1660)
    c.text(70,42,'当前 M4：实际推理架构',46,bold=True)
    c.text(70,108,'E075A step80K · 代码核对于 2026-11-15 · T = 24（历史部署）/ 32（最新主表）',25,MUTED)
    c.box((550,170,700,100),'输入：光学图像 + SAR 图像',['512 × 512，三通道模型输入'],BLUE,size=25)
    c.rect((65,350,800,590),'#ffffff','#9ebfe1')
    c.rect((935,350,800,590),'#ffffff','#8ec9be')
    c.text(100,373,'B 分支：互补匹配器 + 像素门控',31,BLUE,True)
    c.text(970,373,'C 分支：F3 初始化的蒸馏 student',31,TEAL,True)
    c.arrow([(900,270),(900,310),(85,310),(85,420),(465,420)],BLUE,head=False)
    c.arrow([(900,310),(1710,310),(1710,420),(1335,420),(1335,435)],TEAL)
    c.box((105,435,340,136),'F2 特征级融合',['1 套相关金字塔','1 个 GRU 更新器 × T'],BLUE,size=24)
    c.box((485,435,340,136),'F3 相关性级融合',['2 套相关金字塔','1 个 GRU 更新器 × T'],BLUE,size=24)
    c.arrow([(465,420),(275,420),(275,435)],BLUE)
    c.arrow([(465,420),(655,420),(655,435)],BLUE)
    c.arrow([(275,571),(275,610),(465,610),(465,641)],BLUE)
    c.arrow([(655,571),(655,610),(465,610)],BLUE,head=False)
    c.box((110,645,710,174),'B 像素门控：9 通道特征',['读取两支 flow、分歧及 LSR 残差','gB = Conv → Sigmoid','B_t = (1 − gB) · F3_t + gB · F2_t'],BLUE,size=25)
    c.text(465,867,'输出序列 B_1 … B_T',28,BLUE,True,'mt')
    c.arrow([(465,819),(465,854)],BLUE)
    c.box((980,435,710,136),'C student：F3 相关性级融合',['2 套相关金字塔，1 个 GRU 更新器 × T','student 输出直接作为 C_t'],TEAL,'#ecf8f4',24)
    c.text(1335,610,'输出序列 C_1 … C_T',28,TEAL,True,'mt')
    c.arrow([(1335,571),(1335,603)],TEAL)
    c.box((980,720,325,130),'Teacher F2',['仍执行，1 套金字塔','GRU × T'],MUTED,'#edf0f3',22)
    c.box((1365,720,325,130),'Teacher F3',['仍执行，2 套金字塔','GRU × T'],MUTED,'#edf0f3',22)
    c.arrow([(1250,220),(1760,220),(1760,685),(1142,685),(1142,720)],MUTED,True)
    c.arrow([(1527,685),(1527,720)],MUTED,True)
    c.text(1335,878,'灰色为推理冗余：仅生成 teacher_avg 缓存',23,MUTED,anchor='mt')
    # Source output paths deliberately bypass the teacher nodes.
    c.arrow([(465,901),(465,980),(900,980),(900,1025)],PURPLE)
    c.arrow([(1335,650),(1720,650),(1720,980),(900,980)],TEAL,head=False)
    c.box((350,1030,1100,190),'M4 区域门控：融合 B_t 与 C_t',['16 通道：flow、分歧、LSR 投影及投影残差','Conv 16 → 32 → 16 → 1 → Sigmoid → 4×4 平均池化 → 上采样','M4_t = (1 − gM) · B_t + gM · C_t'],PURPLE,'#f4effa',26)
    c.arrow([(900,1220),(900,1270)],AMBER)
    c.box((550,1275,700,115),'选取第 T 轮 → 末端 LSR',['普通最小二乘仿射拟合，200000 点'],AMBER,'#fff5e4',25)
    c.arrow([(900,1390),(900,1430)],AMBER)
    c.box((550,1435,700,100),'输出：仿射约束的稠密匹配位移',['也可表示为 2×3 仿射矩阵或齐次 3×3 矩阵'],TEAL,'#ecf8f4',23)
    c.text(70,1580,'实线：预测依赖；灰线：当前仍执行的冗余计算。各 GRU 的迭代权重共享；LSR 不回写 GRU 状态。',24,MUTED)
    c.save('m4_overall')


def detail():
    c=Canvas(1800,1670)
    c.text(70,42,'展开 F2 / F3：两类融合发生在哪里',44,bold=True)
    c.text(70,108,'每个 F2/F3 实例都有自己的 V1/V2 特征分支；同名不代表跨实例共享权重。',25,MUTED)
    c.box((550,170,700,95),'同一对 OPT / SAR 输入',['归一化至 [−1, 1]'],BLUE,size=25)
    c.arrow([(900,265),(900,300),(455,300),(455,344)],BLUE)
    c.arrow([(900,300),(1345,300),(1345,344)],TEAL)
    c.box((70,350,770,170),'V1：GDROS 风格特征',['模态残差输入适配 → CNN（stride 8）','位置编码 → 跨图 Transformer','输出一对 256 × 64 × 64 特征'],BLUE,size=25)
    c.box((960,350,770,170),'V2：DINO 辅助的 GDROS 风格特征',['GDROS 风格特征 + 冻结 E046 / DINO 前端','DINO 特征投影 / FiLM 残差注入','输出一对 256 × 64 × 64 特征'],TEAL,'#ecf8f4',25)
    c.rect((65,600,800,715),'#ffffff','#9ebfe1')
    c.rect((935,600,800,715),'#ffffff','#8ec9be')
    c.text(100,625,'F2：先融合特征，再计算相关性',31,BLUE,True)
    c.text(970,625,'F3：先分别计算相关性，再融合',31,TEAL,True)
    # Both feature families feed both schematic alternatives.
    c.arrow([(455,520),(455,560),(85,560),(85,680),(465,680),(465,695)],BLUE)
    c.arrow([(1345,520),(1345,560),(1710,560),(1710,680),(1345,680),(1345,687)],TEAL)
    c.arrow([(455,560),(1345,560)],MUTED,head=False)
    c.box((110,695,710,140),'特征门控 + 投影',['分别融合 source / target 的 V1 与 V2 特征','门控为逐位置标量，输出仍为 256 通道'],BLUE,size=25)
    c.arrow([(465,835),(465,884)],BLUE)
    c.box((110,890,710,105),'1 套 4D all-pairs 相关金字塔',['4 层 target 维度平均池化，查询半径 r=4'],BLUE,size=25)
    c.arrow([(465,995),(465,1110)],BLUE)
    c.box((110,1115,710,135),'单个 GRU 更新器，共享权重迭代 T 轮',['相关性查询 + 当前 flow + source context','更新坐标，并学习上采样到 512 × 512'],BLUE,size=25)
    c.box((980,695,325,140),'V1 相关金字塔',['4 层 / r=4','每轮查询 324 通道'],TEAL,'#ecf8f4',23)
    c.box((1365,695,325,140),'V2 相关金字塔',['4 层 / r=4','每轮查询 324 通道'],TEAL,'#ecf8f4',23)
    c.arrow([(1345,687),(1345,690),(1142,690),(1142,695)],TEAL)
    c.arrow([(1345,690),(1527,690),(1527,695)],TEAL)
    c.arrow([(1142,835),(1142,870),(1335,870),(1335,884)],TEAL)
    c.arrow([(1527,835),(1527,870),(1335,870)],TEAL,head=False)
    c.box((980,890,710,175),'每轮融合两套查询响应',['corr_t = (1 − α) · corrV1_t + α · corrV2_t','α 为学习到的全局标量，不是逐对置信度','两套均在同一个当前坐标处查询'],TEAL,'#ecf8f4',24)
    c.arrow([(1335,1065),(1335,1110)],TEAL)
    c.box((980,1115,710,135),'单个 GRU 更新器，共享权重迭代 T 轮',['同一个 recurrent state 接收融合后的响应','两个金字塔不对应两个独立 GRU'],TEAL,'#ecf8f4',25)
    c.box((110,1380,1580,175),'容易混淆的三个概念',['“4D”表示 source (x,y) × target (x,y)，不是 4 个特征尺度。','“4 层金字塔”是每套相关体内部的 target 尺度；F3 有 2 套这样的金字塔。','“T 轮 GRU”是同一更新器复用 T 次；完整 M4 又包含多个独立 matcher 实例。'],PURPLE,'#f4effa',27)
    c.text(70,1602,'LSR 位于外层门控与最终输出；F2/F3 的推理循环内部未进行 LSR 坐标反馈。',25,MUTED)
    c.save('m4_matcher_detail')


if __name__ == '__main__':
    overall()
    detail()
