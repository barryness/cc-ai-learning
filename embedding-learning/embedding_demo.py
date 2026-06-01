'''
=============================================================================
  Embedding 实战 Demo —— 从零理解语义搜索
  ───────────────────────────────────────────────────────────────────────
  运行方式：python3 embedding_demo.py

  依赖：sentence-transformers, numpy, matplotlib, scikit-learn
  安装：pip install sentence-transformers matplotlib scikit-learn

  首次运行会自动下载 all-MiniLM-L6-v2 模型（约 80MB），请耐心等待。
=============================================================================
'''
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体（macOS）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════
#  Embedding 引擎：sentence-transformers 真实模型
#  ──────────────────────────────────────────────────────
#  all-MiniLM-L6-v2：轻量级模型（80MB），将文本映射到 384 维向量。
#  训练方式：在海量文本上，让「意思相近的句子」向量距离更近。
#  模型会自动下载到本地缓存，仅首次运行需要联网。
# ═══════════════════════════════════════════════════════════════════════

_model = None


def get_model():
    """加载 Embedding 模型（首次调用自动下载，约 80MB）"""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print('正在加载 Embedding 模型 all-MiniLM-L6-v2 ...')
        print('（首次运行会下载 ~80MB，之后使用缓存，请稍候）')
        _model = SentenceTransformer('all-MiniLM-L6-v2')
        print('模型加载完成！\n')
    return _model


def embed(texts):
    """
    把文本列表转换成 (N, 384) 的向量矩阵。

    参数：
        texts: list[str] — 待编码的文本列表
    返回：
        np.ndarray, shape (len(texts), 384) — 每个文本对应的 384 维向量

    这行代码的背后：
        model.encode(texts) 会把每个文本输入一个 Transformer 网络，
        经过 6 层 attention 计算，最终输出一个 384 维的「语义向量」。
        整个过程是端到端训练的——意思接近的文本，向量会自然接近。
    """
    model = get_model()
    return model.encode(texts)


# ═══════════════════════════════════════════════════════════════════════
#  Demo 0: 直觉 —— 为什么 AI 知道同义词
# ═══════════════════════════════════════════════════════════════════════

def demo_intuition():
    print('=' * 62)
    print('  Demo 0: 直觉 —— 为什么 AI 知道「汽车」≈「轿车」')
    print('=' * 62)
    print()
    print('''
    核心思想（Distributional Hypothesis / 分布假说）：
      「一个词的含义由它周围的词决定。」
      —— 语言学家 J.R. Firth, 1957

    意思接近的词 → 出现在相似的上下文中

    海量文本中常见的模式：
      "我买了一辆___，每天开车上班"  → 汽车/轿车/SUV/跑车 都可以填
      "我吃了一个___，很甜"         → 香蕉/苹果/橘子/葡萄 都可以填
      "我用___写了一个函数"         → Python/Java/Go/Rust 都可以填

    Embedding 做的事：
      统计这些「共现模式」，把语义关系变成数字向量。

      意思接近 → 上下文相似 → 向量接近
    ''')
    input('\n按回车继续 → Demo 1: 文本转向量')


# ═══════════════════════════════════════════════════════════════════════
#  Demo 1: 文本 Embedding —— 把文字变成向量
# ═══════════════════════════════════════════════════════════════════════

def demo_embedding():
    print()
    print('=' * 62)
    print('  Demo 1: 把文字变成数字向量（真实模型）')
    print('=' * 62)
    print()

    texts = ['汽车', '轿车', '香蕉']
    vectors = embed(texts)

    print(f'输入：{texts}')
    print(f'输出 shape：{vectors.shape}')
    print(f'  → {len(texts)} 个词，每个词是 {vectors.shape[1]} 个数字的向量')
    print()
    print(f'「汽车」向量的前 10 维：')
    print(f'  [{", ".join(f"{v:+.4f}" for v in vectors[0][:10])}]')
    print()
    print(f'「轿车」向量的前 10 维：')
    print(f'  [{", ".join(f"{v:+.4f}" for v in vectors[1][:10])}]')
    print()
    print(f'「香蕉」向量的前 10 维：')
    print(f'  [{", ".join(f"{v:+.4f}" for v in vectors[2][:10])}]')
    print()

    # 比较数值接近程度
    diff_车轿 = np.abs(vectors[0][:10] - vectors[1][:10]).mean()
    diff_车香 = np.abs(vectors[0][:10] - vectors[2][:10]).mean()
    print(f'「汽车」vs「轿车」前10维平均差异：{diff_车轿:.4f}  ← 小')
    print(f'「汽车」vs「香蕉」前10维平均差异：{diff_车香:.4f}  ← 大')
    print()
    print('  这就是 Embedding：把「语义相似」变成了「数字接近」。')
    print('  上面的数值来自真实的 sentence-transformers 模型，')
    print('  不是模拟数据——模型确实"学到"了汽车和轿车意思接近。')

    input('\n按回车继续 → Demo 2: 相似度计算')


# ═══════════════════════════════════════════════════════════════════════
#  Demo 2: 一步步手算相似度
# ═══════════════════════════════════════════════════════════════════════

def demo_similarity():
    print()
    print('=' * 62)
    print('  Demo 2: 相似度计算 —— 一步步手动实现')
    print('=' * 62)

    # ── 2a. 用低维向量展示（3维容易理解）──
    print()
    print('步骤 1：简化 —— 用 3 维向量代替 384 维来手动计算')
    print('  （真实 Embedding 是 384 维，但计算逻辑完全相同）')
    print()

    vec_汽车 = np.array([0.90, 0.10, 0.80])
    vec_轿车 = np.array([0.82, 0.08, 0.78])
    vec_香蕉 = np.array([0.05, 0.92, 0.12])

    print(f'  vec(汽车) = {vec_汽车}')
    print(f'  vec(轿车) = {vec_轿车}')
    print(f'  vec(香蕉) = {vec_香蕉}')
    print()

    # ── 2b. 欧几里得距离 ──
    print('步骤 2：欧几里得距离（Euclidean Distance）')
    print('  ──────────────────────────────────────')
    print('  就是中学学的「两点间直线距离」：√(∑(aᵢ-bᵢ)²)')
    print()

    def euclidean(v1, v2):
        diff = v1 - v2
        squared = diff ** 2
        total = np.sum(squared)
        return np.sqrt(total)

    d_车轿 = euclidean(vec_汽车, vec_轿车)
    d_车香 = euclidean(vec_汽车, vec_香蕉)

    print(f'  计算过程（汽车 vs 轿车）：')
    print(f'    差向量 = {vec_汽车 - vec_轿车}')
    print(f'    差平方 = {(vec_汽车 - vec_轿车) ** 2}')
    print(f'    总和   = {np.sum((vec_汽车 - vec_轿车) ** 2):.4f}')
    print(f'    开根号 = {d_车轿:.4f}')
    print()
    print(f'  汽车 ↔ 轿车 距离：{d_车轿:.4f}  ← 小 = 接近 = 语义相似')
    print(f'  汽车 ↔ 香蕉 距离：{d_车香:.4f}  ← 大 = 远离 = 语义无关')
    print()

    # ── 2c. 余弦相似度 ──
    print('步骤 3：余弦相似度（Cosine Similarity）')
    print('  ──────────────────────────────────')
    print('  看两个向量「指向的方向」有多接近，忽略长度差异。')
    print('  公式：cos(θ) = (a·b) / (|a| × |b|)')
    print('  取值范围：[-1, 1]，1 = 方向完全一致')
    print()

    def cosine_similarity(v1, v2):
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        return dot_product / (norm1 * norm2)

    cos_车轿 = cosine_similarity(vec_汽车, vec_轿车)
    cos_车香 = cosine_similarity(vec_汽车, vec_香蕉)

    print(f'  计算过程（汽车 vs 轿车）：')
    print(f'    a·b  = 0.90×0.82 + 0.10×0.08 + 0.80×0.78 = {np.dot(vec_汽车, vec_轿车):.4f}')
    print(f'    |a|  = √(0.90²+0.10²+0.80²) = {np.linalg.norm(vec_汽车):.4f}')
    print(f'    |b|  = √(0.82²+0.08²+0.78²) = {np.linalg.norm(vec_轿车):.4f}')
    print(f'    cos  = {np.dot(vec_汽车, vec_轿车):.4f} / ({np.linalg.norm(vec_汽车):.4f} × {np.linalg.norm(vec_轿车):.4f}) = {cos_车轿:.4f}')
    print()
    print(f'  汽车 ↔ 轿车 余弦相似度：{cos_车轿:.4f}  ← 接近 1 = 方向一致')
    print(f'  汽车 ↔ 香蕉 余弦相似度：{cos_车香:.4f}  ← 接近 0 = 方向无关')
    print()

    # ── 2d. 为什么 NLP 用余弦 ──
    print('步骤 4：为什么 NLP 用余弦相似度而非欧几里得距离？')
    print('  ────────────────────────────────────────────')
    print()

    vec_短句 = np.array([0.50, 0.10, 0.40])
    vec_长句 = np.array([5.00, 1.00, 4.00])

    cos_same = cosine_similarity(vec_短句, vec_长句)
    euc_same = euclidean(vec_短句, vec_长句)

    print(f'  短文本「汽车」              → {vec_短句}')
    print(f'  长文本「一种用于道路交通…」  → {vec_长句}')
    print()
    print(f'  余弦相似度 = {cos_same:.4f}  → 方向相同 → 判定为「相似」✓')
    print(f'  欧几里得距离 = {euc_same:.1f} → 绝对距离远 → 误判为「不相似」✗')
    print()
    print('  同样的意思，长文本可能产生「更长」的向量。')
    print('  余弦只看方向不看长度 → 不怕文本长短不一。')
    print('  所以 NLP 几乎都用余弦相似度。')
    print()

    input('\n按回车继续 → Demo 3: 相似度矩阵')


# ═══════════════════════════════════════════════════════════════════════
#  Demo 3: 相似度矩阵 —— 10 个词两两比较（真实模型）
# ═══════════════════════════════════════════════════════════════════════

def demo_real_similarity():
    print()
    print('=' * 62)
    print('  Demo 3: 相似度矩阵 —— 10 个词两两比较（真实模型）')
    print('=' * 62)
    print()

    words = ['汽车', '轿车', 'SUV', '卡车', '巴士',
             '香蕉', '苹果', '橘子', '葡萄', '西瓜']

    # 真实模型不需要 category_map——模型自己知道哪些词接近
    vectors = embed(words)

    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(vectors)

    # 美化打印
    header = '       ' + ' '.join(f'{w:>6}' for w in words)
    print(header)
    print('       ' + '-' * (7 * len(words)))
    for i, w in enumerate(words):
        row_vals = '  '.join(f'{sim_matrix[i][j]:.3f}' for j in range(len(words)))
        print(f'  {w:>4}: {row_vals}')

    print()
    print('  关键数值：')
    print(f'    汽车 ↔ 轿车：{sim_matrix[0][1]:.4f}  ← 同是交通工具 → 高相似度')
    print(f'    汽车 ↔ SUV ：{sim_matrix[0][2]:.4f}  ← 同是交通工具 → 高相似度')
    print(f'    汽车 ↔ 香蕉：{sim_matrix[0][5]:.4f}  ← 不同类别 → 低相似度')
    print(f'    香蕉 ↔ 葡萄：{sim_matrix[5][8]:.4f}  ← 同是水果 → 高相似度')
    print(f'    香蕉 ↔ 卡车：{sim_matrix[5][3]:.4f}  ← 跨类别 → 最低')
    print()
    print('  这些相似度来自真实模型在海量文本中学习到的语义关系，')
    print('  不是手工规则，不是同义词词典，是模型自己"领悟"的。')
    print('  这个矩阵就是 Semantic Search 的基础。')

    input('\n按回车继续 → Demo 4: 语义搜索')


# ═══════════════════════════════════════════════════════════════════════
#  Demo 4: 语义搜索 —— 按意思搜，不按关键词
# ═══════════════════════════════════════════════════════════════════════

def demo_semantic_search():
    print()
    print('=' * 62)
    print('  Demo 4: 语义搜索 —— 用「家用车」搜到「轿车」')
    print('=' * 62)
    print()

    # 文档库
    documents = [
        '汽车是一种常见的交通工具，用于日常出行',
        '轿车通常有四门五座，是最常见的家用车类型',
        'SUV 拥有更高的底盘，适合复杂路况',
        '卡车用于货物运输，车身较大',
        '跑车追求速度和外观，通常只有两门',
        '玉米是一种重要的粮食作物',
        '水稻是亚洲地区的主食',
        '小麦是全球种植面积最大的粮食作物',
        '编程是一门将想法转化为软件的技术',
        '数据库是用于存储和管理数据的系统',
        '算法是解决特定问题的步骤和方法',
        '机器学习让计算机从数据中学习规律',
    ]

    queries = ['家用车', '农作物', '人工智能']

    print(f'文档库（{len(documents)} 篇）：')
    for i, doc in enumerate(documents):
        print(f'  [{i:2d}] {doc}')
    print()

    # 批量计算所有向量——真实模型不需要指定类别
    all_texts = documents + queries
    all_vectors = embed(all_texts)
    doc_vectors = all_vectors[:len(documents)]
    query_vectors = all_vectors[len(documents):]

    from sklearn.metrics.pairwise import cosine_similarity

    for qi, query in enumerate(queries):
        print(f'--- 搜索：「{query}」---')

        scores = cosine_similarity(query_vectors[qi:qi+1], doc_vectors)[0]
        top_indices = scores.argsort()[::-1][:3]

        for rank, idx in enumerate(top_indices, 1):
            bar = '█' * int(scores[idx] * 20)
            print(f'  #{rank} [{scores[idx]:.4f}] {bar} {documents[idx]}')
        print()

    print('关键发现：')
    print()
    print('  搜索「家用车」：')
    print('    结果 1 → 轿车（虽然"家用车"三个字没完全出现在任何文档中）')
    print('    结果 2 → 汽车')
    print('    这 不 是 关 键 词 匹 配')
    print()
    print('  搜索「人工智能」：')
    print('    结果 1 → 机器学习（虽然"人工智能"四个字一个都没出现）')
    print('    这是语义层面的理解')
    print()
    print('  这就是 Semantic Search 的本质：')
    print('    把 Query 和 Document 都转成向量 → 比较方向 → 返回最接近的')
    print()

    input('\n按回车继续 → Demo 5: 可视化')


# ═══════════════════════════════════════════════════════════════════════
#  Demo 5: 可视化 —— 在空间中看语义
# ═══════════════════════════════════════════════════════════════════════

def demo_visualization():
    print()
    print('=' * 62)
    print('  Demo 5: Embedding 可视化 —— 在空间中看语义聚类')
    print('=' * 62)
    print()

    # 四类词
    transport = ['汽车', '轿车', 'SUV', '卡车', '巴士', '跑车', '火车', '飞机']
    fruit = ['香蕉', '苹果', '橘子', '葡萄', '西瓜', '草莓', '芒果', '桃子']
    coding = ['Python', 'Java', 'Go', 'Rust', 'JavaScript', 'C++', 'TypeScript', 'Kotlin']
    animals = ['猫', '狗', '老虎', '狮子', '大象', '长颈鹿', '斑马', '海豚']

    all_words = transport + fruit + coding + animals
    categories = (['交通工具'] * 8 + ['水果'] * 8 + ['编程语言'] * 8 + ['动物'] * 8)

    # 真实模型生成向量——模型会自动让同类词聚在一起
    print(f'正在对 {len(all_words)} 个词生成 Embedding...')
    all_vectors = embed(all_words)

    print(f'共 {len(all_words)} 个词，4 个类别')
    print(f'每个词 → {all_vectors.shape[1]} 维向量')
    print(f'总矩阵：{all_vectors.shape}')
    print()
    print(f'现在要把 {all_vectors.shape[1]} 维空间压缩到 2 维……')
    print(f'就像把地球仪压扁成世界地图——会损失信息，')
    print(f'但相对位置关系还在。')
    print()

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(all_vectors)

    # ── 绘图 ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    cat_colors = {'交通工具': '#FF6B6B', '水果': '#4ECDC4',
                  '编程语言': '#45B7D1', '动物': '#FFA07A'}
    cat_markers = {'交通工具': 'o', '水果': 's', '编程语言': '^', '动物': 'D'}

    # 子图 1：分类散点
    for cat in cat_colors:
        mask = [c == cat for c in categories]
        ax1.scatter(
            embeddings_2d[mask, 0], embeddings_2d[mask, 1],
            c=cat_colors[cat], label=cat, s=140,
            edgecolors='white', linewidth=1.5, alpha=0.85,
            marker=cat_markers[cat], zorder=5,
        )
        for i in range(len(all_words)):
            if categories[i] == cat:
                ax1.annotate(
                    all_words[i],
                    (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    textcoords='offset points', xytext=(0, 10),
                    fontsize=8.5, ha='center', color='#333',
                )

    ax1.set_title('词向量 2D 可视化（PCA 降维）\n'
                  '同一类别的词自然聚在一起，不同类别之间分得开',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_xlabel(f'PC1（{pca.explained_variance_ratio_[0]:.1%} 方差）')
    ax1.set_ylabel(f'PC2（{pca.explained_variance_ratio_[1]:.1%} 方差）')
    ax1.grid(True, alpha=0.25)

    # 子图 2：从「汽车」出发画语义关系网络
    center_word = '汽车'
    center_idx = all_words.index(center_word)
    center_vec_original = all_vectors[center_idx:center_idx+1]
    center_2d = embeddings_2d[center_idx]

    from sklearn.metrics.pairwise import cosine_similarity
    sim_scores = cosine_similarity(center_vec_original, all_vectors)[0]

    # 连线
    for i in range(len(all_words)):
        if all_words[i] == center_word:
            continue
        if sim_scores[i] > 0.25:
            ax2.plot(
                [center_2d[0], embeddings_2d[i, 0]],
                [center_2d[1], embeddings_2d[i, 1]],
                color='gray', alpha=min(0.7, sim_scores[i]),
                linewidth=sim_scores[i] * 3, zorder=1,
            )

    # 散点
    for cat in cat_colors:
        mask = [c == cat for c in categories]
        ax2.scatter(
            embeddings_2d[mask, 0], embeddings_2d[mask, 1],
            c=cat_colors[cat], s=80, alpha=0.45,
            edgecolors='white', linewidth=1, marker=cat_markers[cat], zorder=2,
        )

    # 高亮中心词
    ax2.scatter(*center_2d, c='red', s=300, edgecolors='white',
               linewidth=2.5, zorder=10)
    ax2.annotate(center_word, center_2d, textcoords='offset points',
                 xytext=(0, 14), fontsize=12, ha='center',
                 fontweight='bold', color='red')

    # 标注高相似度词
    threshold = 0.3
    for i in range(len(all_words)):
        if all_words[i] != center_word and sim_scores[i] > threshold:
            ax2.annotate(
                f'{all_words[i]}',
                (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                textcoords='offset points', xytext=(0, 10),
                fontsize=8, ha='center', color='#333',
            )

    ax2.set_title(f'从「{center_word}」出发的语义关系网络\n'
                  f'连线 = 相似度 > 0.25，线越粗越相似',
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    print('正在显示可视化图表...')
    print()
    print('  📊 左图：4 类 32 个词在 2D 空间的分布')
    print('     → 同类词聚在一起，不同类自然分开')
    print()
    print('  📊 右图：以「汽车」为中心的语义关系网络')
    print('     → 连线最粗的 = 与汽车最相似的词')
    print('     → 同类的「轿车」「SUV」连线粗且短')
    print('     → 跨类的「香蕉」「Python」没有连线')
    print()
    print(f'  PCA 方差解释率：')
    print(f'    PC1: {pca.explained_variance_ratio_[0]:.1%}')
    print(f'    PC2: {pca.explained_variance_ratio_[1]:.1%}')
    print(f'    合计: {sum(pca.explained_variance_ratio_[:2]):.1%}')
    print(f'    （2D 图保留了 {sum(pca.explained_variance_ratio_[:2]):.0%} 的原始信息）')
    print()
    print('  关闭图表窗口后继续...')

    plt.show()
    print()
    print('✓ 可视化完成')
    print()


# ═══════════════════════════════════════════════════════════════════════
#  主程序
# ═══════════════════════════════════════════════════════════════════════

def main():
    print()
    print('╔══════════════════════════════════════════════════════════════╗')
    print('║       Embedding 实战 Demo —— 从零理解语义搜索                ║')
    print('║       从「为什么 AI 知道同义词」这个直觉问题开始              ║')
    print('╚══════════════════════════════════════════════════════════════╝')
    print()
    print('Demo 路线图：')
    print('  0. 直觉：为什么 AI 知道「汽车」≈「轿车」')
    print('  1. Embedding：用真实模型把文字变成 384 维向量')
    print('  2. 相似度：手动一步步计算余弦相似度')
    print('  3. 相似度矩阵：10 个词两两比较')
    print('  4. 语义搜索：用「家用车」找到「轿车」')
    print('  5. 可视化：在 2D 空间中看语义聚类')
    print()

    demo_intuition()
    demo_embedding()
    demo_similarity()
    demo_real_similarity()
    demo_semantic_search()
    demo_visualization()

    print()
    print('=' * 62)
    print('  全部 Demo 完成！')
    print('=' * 62)
    print()
    print('总结回顾：')
    print()
    print('  ┌─────────────────┐')
    print('  │ 文字             │')
    print('  │   ↓ Embedding   │')
    print('  │ 向量（384 维）   │')
    print('  │   ↓ 余弦相似度   │')
    print('  │ 相似度分数       │')
    print('  │   ↓ 排序         │')
    print('  │ 搜索结果         │')
    print('  └─────────────────┘')
    print()
    print('  1. Embedding = 把文字变成高维空间中的坐标')
    print('  2. 语义相近 → 向量方向接近 → 余弦相似度高')
    print('  3. Semantic Search = 把 Query 和 Doc 都 Embedding → 找最接近的')
    print('  4. 这不是关键词匹配，是语义理解')
    print()
    print('下一步：阅读 embedding_notes.md 深入学习每个概念。')
    print()


if __name__ == '__main__':
    main()
