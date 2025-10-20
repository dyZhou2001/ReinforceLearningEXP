import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']  # 任选其一可用
plt.rcParams['mathtext.fontset'] = 'dejavusans'  # 用 DejaVu 渲染数学符号
plt.rcParams['axes.unicode_minus'] = False       # 负号

# Helper function to draw a box with text
def draw_box(ax, xy, text, width=2.8, height=0.8, fontsize=9):
    x, y = xy
    rect = mpatches.FancyBboxPatch(
        (x, y - height), width, height,
        boxstyle="round,pad=0.2", ec="black", fc="#DCE6F1"
    )
    ax.add_patch(rect)
    ax.text(x + width/2, y - height/2, text,
            ha="center", va="center", fontsize=fontsize, wrap=True)
    return rect

# Create figure
fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis("off")

# Y positions (from top to bottom)
y = 11
dy = 1.5

# Draw boxes for the training step pipeline
boxes = {}

boxes["sample"] = draw_box(ax, (3.5, y), "采样 batch\n(s, a, Rⁿ, s', done)\n\ns:(B, state_dim)\na:(B, action_dim)", width=3.5)
y -= dy

boxes["target_actor"] = draw_box(ax, (3.5, y), "Target Actor μ̄(s')\n输出: a' (B, action_dim)", width=3.5)
y -= dy

boxes["target_critic"] = draw_box(ax, (3.5, y), "Target Critic Z̄(s',a')\n输出: p̃ (B, N)", width=3.5)
y -= dy

boxes["projection"] = draw_box(ax, (3.5, y), "分布投影\n输入: p̃, Rⁿ, done\n输出: m (B, N)", width=3.5)
y -= dy

boxes["critic"] = draw_box(ax, (3.5, y), "Critic Zθ(s,a)\n输出: pθ (B, N)", width=3.5)
y -= dy

boxes["critic_loss"] = draw_box(ax, (3.5, y), "Critic 损失\nLcritic = -Σ m log pθ\n标量", width=3.5)
y -= 2

# Actor branch (parallel)
y_actor = 8.5
boxes["actor"] = draw_box(ax, (0.5, y_actor), "Actor μϕ(s)\n输出: a (B, action_dim)", width=3.2)
y_actor -= dy

boxes["actor_critic"] = draw_box(ax, (0.5, y_actor), "Critic Zθ(s, μϕ(s))\n输出: pθ (B, N)", width=3.2)
y_actor -= dy

boxes["actor_q"] = draw_box(ax, (0.5, y_actor), "期望Q = Σ z_i pθ,i\n输出: Q (B,)", width=3.2)
y_actor -= dy

boxes["actor_loss"] = draw_box(ax, (0.5, y_actor), "Actor 损失\nLactor = -mean(Q)\n标量", width=3.2)

# Soft update at the bottom
boxes["update"] = draw_box(ax, (3.5, 0.5), "软更新 target nets\nθ̄ ← τθ+(1-τ)θ̄\nϕ̄ ← τϕ+(1-τ)ϕ̄", width=3.5)

# Draw arrows
def connect(ax, box1, box2, pos="mid"):
    b1 = box1.get_bbox()
    b2 = box2.get_bbox()
    if pos=="mid":
        x1, y1 = b1.x0 + b1.width/2, b1.y0
        x2, y2 = b2.x0 + b2.width/2, b2.y1
    elif pos=="right":
        x1, y1 = b1.x1, b1.y0 + b1.height/2
        x2, y2 = b2.x0, b2.y1 + b2.height/2
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.5))

# Main chain arrows
connect(ax, boxes["sample"], boxes["target_actor"])
connect(ax, boxes["target_actor"], boxes["target_critic"])
connect(ax, boxes["target_critic"], boxes["projection"])
connect(ax, boxes["projection"], boxes["critic"])
connect(ax, boxes["critic"], boxes["critic_loss"])
connect(ax, boxes["critic_loss"], boxes["update"])

# Actor branch arrows
connect(ax, boxes["actor"], boxes["actor_critic"])
connect(ax, boxes["actor_critic"], boxes["actor_q"])
connect(ax, boxes["actor_q"], boxes["actor_loss"])
connect(ax, boxes["actor_loss"], boxes["update"], pos="right")

plt.tight_layout()
plt.show()
