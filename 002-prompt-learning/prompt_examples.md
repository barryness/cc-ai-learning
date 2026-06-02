# Prompt Engineering 案例集

> 每个案例展示同一个问题在 4 个 Prompt 版本下的输出差异。
> 所有输出均为真实 API 调用结果（DeepSeek V4 Flash，temperature=0.3）。

---

## 案例 1：SQL 生成

### 用户需求
> 帮我写一个SQL

---

### V1：裸提示词（零要素）

```python
messages = [HumanMessage(content="帮我写一个SQL")]
```

**LLM 输出**：

> 你提供的需求非常简略，仅写了"帮我写一个SQL"，没有说明具体的表结构、查询目标或数据库类型。
>
> 为了能准确帮助你，我需要你补充以下信息：
> 1. 要执行什么操作？（查询、插入、更新、删除、建表等）
> 2. 涉及哪些表？（表名、字段名、字段类型、主键/外键关系）
> 3. 具体的业务逻辑或筛选条件是什么？
>
> 这里给你一个通用的 SELECT 查询示例：
> ```sql
> SELECT id, username, email, age, created_at
> FROM users
> WHERE age > 18
> ORDER BY created_at DESC;
> ```

**问题诊断**：LLM 完全不知道你要什么。它在所有可能的方向中取平均 → 输出 = 追问 + 无意义的通用示例。**不可用**。

---

### V2：+Role

```python
messages = [
    SystemMessage(content="你是一个资深数据仓库工程师，精通SQL。"),
    HumanMessage(content="帮我写一个SQL"),
]
```

**LLM 输出**：

> 先给您一个数据仓库中常见的**月度销售汇总**示例（假设有订单表 sales 和日期维度表 dim_date）：
>
> ```sql
> WITH daily_sales AS (
>     SELECT dim_date.year, dim_date.month,
>            SUM(sales.amount) AS total_amount,
>            COUNT(DISTINCT sales.order_id) AS order_cnt
>     FROM fact_sales AS sales
>     INNER JOIN dim_date ON sales.date_key = dim_date.date_key
>     WHERE dim_date.year = 2024
>     GROUP BY dim_date.year, dim_date.month
> )
> SELECT year, month, total_amount, order_cnt, ...
> ```

**改善**：角色起作用了。LLM 给出了"数据仓库风格"的 SQL——CTE、维度表 JOIN、聚合函数。不再是通用的 `SELECT * FROM users`。

**问题**：仍然是**凭空猜测**表结构（`fact_sales`、`dim_date` 完全不存在于你的系统）。

---

### V3：+Role +Context +Constraint

```python
messages = [
    SystemMessage(content="""你是一个资深数据仓库工程师，精通SQL性能优化。

## 上下文
- 数据库：MySQL 8.0
- 场景：订单数据仓库，表名 orders_dw
- 字段：order_id, customer_id, product_id, amount, order_date, status

## 约束
1. 必须使用标准SQL
2. 给出索引建议
3. 解释为什么这样写
4. 不超过300字"""),
    HumanMessage(content="帮我写一个SQL"),
]
```

**LLM 输出**：

```sql
SELECT
    customer_id,
    SUM(amount) AS total_amount
FROM orders_dw
WHERE order_date >= CURDATE() - INTERVAL 90 DAY
GROUP BY customer_id
ORDER BY total_amount DESC;
```

> **索引建议**：创建复合索引 `(order_date, customer_id, amount)`。
> **原因**：
> - 该查询先通过 `order_date` 过滤，索引的第一列可以快速定位最近90天的数据。
> - 后续的 `GROUP BY customer_id` 利用索引的第二列完成分组，避免排序或临时表。
> - `amount` 作为索引第三列，可使索引覆盖查询（无需回表获取 amount 值）。

**质变**：
- 用了真实的表名 `orders_dw` 和字段名 ✓
- WHERE 条件用了真实的字段 `order_date`、`status` ✓
- 给了索引建议和性能解释 ✓
- Context 和 Constraint 的组合产生了**可直接使用的输出**

---

### V4：+Role +Context +Constraint +Example +Schema

```python
messages = [
    SystemMessage(content="""你是一个资深数据仓库工程师，精通SQL性能优化。

## 上下文
- 数据库：MySQL 8.0
- 场景：订单数据仓库，表名 orders_dw
- 字段：order_id, customer_id, product_id, amount, order_date, status

## 约束
1. 标准SQL，避免方言语法
2. 给出索引建议
3. 解释关键逻辑
4. 不超过300字

## 参考示例（参照此风格）
```sql
-- 需求：最近30天各品类销售额Top 10
-- 思路：先按品类汇总，再通过窗口函数排名
SELECT category_id, total_amount, rnk FROM (
    SELECT category_id,
           SUM(amount) AS total_amount,
           ROW_NUMBER() OVER (ORDER BY SUM(amount) DESC) AS rnk
    FROM orders_dw
    WHERE order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 30 DAY)
      AND status = 'completed'
    GROUP BY category_id
) t WHERE rnk <= 10;
-- 索引建议：(order_date, status, category_id) 联合索引
```

## 输出格式
- 需求理解：
- SQL：
- 思路说明：
- 索引建议："""),
    HumanMessage(content="帮我写一个SQL"),
]
```

**LLM 输出**：

> ### 需求理解
> 统计最近一年每月已完成订单的销售额，并找出销售额最高的前10个月份，按月份升序展示。
>
> ### SQL
> ```sql
> SELECT month, total_amount, rnk
> FROM (
>     SELECT DATE_FORMAT(order_date, '%Y-%m') AS month,
>            SUM(amount) AS total_amount,
>            ROW_NUMBER() OVER (ORDER BY SUM(amount) DESC) AS rnk
>     FROM orders_dw
>     WHERE order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 1 YEAR)
>       AND status = 'completed'
>     GROUP BY DATE_FORMAT(order_date, '%Y-%m')
> ) t WHERE rnk <= 10
> ORDER BY month;
> ```
>
> ### 思路说明
> 1. 筛选最近一年且状态为 completed 的订单（过滤无效数据）
> 2. 按年月分组，计算每月销售额总和
> 3. 使用窗口函数 ROW_NUMBER() 按销售额降序排名，取前10名
> 4. 外层查询按月份排序
>
> ### 索引建议
> (order_date, status) 复合索引覆盖 WHERE 过滤，后续按月份聚合

**专业级输出**：
- 结构严整，严格遵循 Output Schema ✓
- 代码带注释，风格完全模仿 Example ✓
- 思路分层清晰 ✓
- 索引建议精确 ✓
- **可以直接贴到生产环境** ✓

---

### V1→V4 效果对比表

| 维度 | V1 裸提示 | V2 +Role | V3 +Ctx+Cst | V4 +Ex+Sch |
|------|----------|----------|-------------|------------|
| 用真实表名 | ✗ | ✗ | ✓ | ✓ |
| SQL 可直接运行 | ✗ | ✗ | ✓ | ✓ |
| 有索引建议 | ✗ | ✗ | ✓ | ✓ |
| 逻辑分层解释 | ✗ | ✗ | 简要 | 清晰 |
| 输出结构固定 | ✗ | ✗ | ✗ | ✓ |
| 代码有注释 | ✗ | ✗ | ✗ | ✓ |
| 风格一致性 | ✗ | ✗ | 一般 | 优秀 |
| 生产就绪 | ✗ | ✗ | 基本 | 完全 |

---

## 案例 2：技术文档

### 用户需求
> 写一个 Redis 的使用文档

### V1 裸提示 vs V4 完整 Prompt

**V1 输出**（节选）：
> Redis 是一个开源的内存数据结构存储系统，可以用作数据库、缓存和消息代理...

**V4 Prompt**：
```markdown
你是 Redis 核心贡献者，参与过 Redis Cluster 的开发。

## 上下文
- 读者：有 3 年 Java 经验的后端工程师，熟悉数据库但没深入用过 Redis
- 场景：新项目要用 Redis 做缓存和分布式锁，数据量预估 100 万 key

## 约束
1. 每个功能给出"最佳实践"和"常见踩坑"两条
2. 包含代码示例（Java + Jedis）
3. 不要讲 Redis 的安装步骤

## 参考示例
```java
// Redis 缓存穿透的解决方案 —— 布隆过滤器
BloomFilter<String> filter = BloomFilter.create(
    Funnels.stringFunnel(Charset.defaultCharset()),
    1000000, 0.01);
if (!filter.mightContain(key)) return null; // 一定不存在
String value = jedis.get(key);
if (value == null) {
    value = db.query(key);
    jedis.setex(key, 3600, value);
}
```

## 输出格式
- 需求场景：
- 最佳实践：
- 常见踩坑：
- 代码示例：
```

**V4 输出特点**：
- 直接跳过安装，切入使用场景 ✓
- 每个功能都有"最佳实践"+"踩坑" ✓
- Java + Jedis 代码可运行 ✓
- 结构完全按照 Output Schema ✓

---

## 案例 3：Bug 诊断

### 用户需求
> 系统突然变慢了，帮我看看

### V4 Prompt
```markdown
你是阿里巴巴中间件团队的资深性能诊断专家，排查过 1000+ 线上故障。

## 上下文
- 系统：Spring Boot 微服务，MySQL 5.7
- 现象：昨天部署后，P99 延迟从 200ms 飙升到 3s
- 变更：新增了一个查询接口
- 日志：[附 Slow Query Log 片段]

## 约束
1. 按优先级排查：硬件 → 连接池 → SQL → 索引 → 锁
2. 每步给出"怎么看"的具体命令
3. 不要建议"加机器"这类规避方案

## 输出格式
- 最可能的原因（Top 3）：
- 诊断步骤：
- 立即止血方案：
- 长期修复方案：
```

---

## 五要素速查表

| 要素 | 作用 | 何时用 | 何时可以省 |
|------|------|--------|-----------|
| Role | 激活专业领域 | 需要专业判断时 | 通用问答 |
| Context | 限定信息范围 | 有特定业务场景时 | 无关背景的通用问题 |
| Constraint | 排除错误输出 | 有明确的"不要"时 | 开放式探索 |
| Example | 示范期望模式 | 输出质量要求高时 | 格式自由的任务 |
| Schema | 控制输出结构 | 需要解析或自动处理时 | 给人类看的非结构化内容 |

**经验法则**：
- 简单问答 → Role 就够了
- 代码生成 → Role + Context + Constraint（三件套）
- 生产级输出 → 五要素全上 + 1 个好 Example
