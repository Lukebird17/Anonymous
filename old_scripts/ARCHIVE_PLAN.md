# 📦 项目文件整理方案

## 🌟 核心文件 - 必须保留

### 最新演示网页（✅ 保留）
```
results/animated_attack_demo.html          ⭐ 最终版演示页面
results/real_data_demo.json                ⭐ 演示数据文件
results/test_highlight.html                ⭐ 测试页面
```

### 最新代码（✅ 保留）
```
main_experiment_unified.py                 ⭐ 统一实验入口
generate_real_demo_data.py                 ⭐ 数据生成脚本
visualize_interactive_dashboard.py         ⭐ 最新可视化代码（如果用到）

attack/                                     ⭐ 攻击算法
defense/                                    ⭐ 防御算法
models/                                     ⭐ 模型代码
preprocessing/                              ⭐ 预处理
utils/                                      ⭐ 工具函数
data/                                       ⭐ 数据集
```

---

## 🗂️ 旧文件 - 移到 OLD_VERSIONS/

### 旧的可视化脚本（❌ 归档）
```
visualize_html.py
visualize_all_unified.py
visualize_attack_principles.py
visualize_attack_principles_v2.py
visualize_complete_template.py
visualize_principles_complete.py
visualize_unified_auto.py
```

### 旧的HTML页面（❌ 归档）
```
results/attack_complete.html                # 模拟数据版本
results/attack_principles_demo.html
results/attack_demo_improved.html
results/attack_standalone.html
results/attack_animation.html
results/real_ego_enhanced_demo.html
results/animated_attack_demo.html.bak
results/animated_attack_demo_test.html
results/stage1_complete_viz.html
results/test_d3.html
results/test_load.html
```

### 旧的数据文件（❌ 归档）
```
results/all_stages_demo_data.json
results/animated_demo_data.json
results/stage1_demo_data.json
results/test_demo.json
results/__demo_20260102_204544.json
results/real_ego_demo_data.json
```

### 旧的文档（❌ 归档）
```
ANIMATION_IMPROVEMENTS.md
ANIMATION_UPDATE_V3.md
BUGFIX_DATA_LOADING.md
DEMO_PREVIEW.md
ENHANCED_DEMO_GUIDE.md
IMPROVED_DEMO_GUIDE.md
LIVE_SERVER_FIX.md
REAL_DATA_GUIDE.md
REAL_DATA_QUICK_START.md
STATS_FIX.md
TASK_METHOD_MAPPING.md
VISUALIZATION_GUIDE.md
COLOR_ATTRIBUTE_COMPLETE.md
COMPLETION_SUMMARY.md
HIGHLIGHT_FIX_README.md
```

### 旧的脚本（❌ 归档）
```
complete_restart.sh
diagnose_issue.sh
fix_browser_cache.sh
generate_demo_from_results.sh
quick_fix.sh
run_*.sh （除了最常用的）
generate_all_stages_data.py
generate_stage1_data.py
```

---

## 📝 执行命令

```bash
cd /home/honglianglu/hdd/Anonymous

# 1. 创建归档目录
mkdir -p OLD_VERSIONS/{scripts,html,data,docs,shell_scripts}

# 2. 归档旧的可视化脚本
mv visualize_html.py OLD_VERSIONS/scripts/
mv visualize_all_unified.py OLD_VERSIONS/scripts/
mv visualize_attack_principles.py OLD_VERSIONS/scripts/
mv visualize_attack_principles_v2.py OLD_VERSIONS/scripts/
mv visualize_complete_template.py OLD_VERSIONS/scripts/
mv visualize_principles_complete.py OLD_VERSIONS/scripts/
mv visualize_unified_auto.py OLD_VERSIONS/scripts/

# 3. 归档旧的HTML
mv results/attack_complete.html OLD_VERSIONS/html/
mv results/attack_principles_demo.html OLD_VERSIONS/html/
mv results/attack_demo_improved.html OLD_VERSIONS/html/
mv results/attack_standalone.html OLD_VERSIONS/html/
mv results/attack_animation.html OLD_VERSIONS/html/
mv results/real_ego_enhanced_demo.html OLD_VERSIONS/html/
mv results/animated_attack_demo.html.bak OLD_VERSIONS/html/
mv results/animated_attack_demo_test.html OLD_VERSIONS/html/
mv results/stage1_complete_viz.html OLD_VERSIONS/html/
mv results/test_d3.html OLD_VERSIONS/html/
mv results/test_load.html OLD_VERSIONS/html/

# 4. 归档旧的数据
mv results/all_stages_demo_data.json OLD_VERSIONS/data/
mv results/animated_demo_data.json OLD_VERSIONS/data/
mv results/stage1_demo_data.json OLD_VERSIONS/data/
mv results/test_demo.json OLD_VERSIONS/data/
mv results/__demo_20260102_204544.json OLD_VERSIONS/data/
mv results/real_ego_demo_data.json OLD_VERSIONS/data/

# 5. 归档旧的shell脚本
mv complete_restart.sh OLD_VERSIONS/shell_scripts/ 2>/dev/null || true
mv diagnose_issue.sh OLD_VERSIONS/shell_scripts/ 2>/dev/null || true
mv fix_browser_cache.sh OLD_VERSIONS/shell_scripts/ 2>/dev/null || true
mv generate_demo_from_results.sh OLD_VERSIONS/shell_scripts/ 2>/dev/null || true
mv quick_fix.sh OLD_VERSIONS/shell_scripts/ 2>/dev/null || true
mv run_demo.sh OLD_VERSIONS/shell_scripts/ 2>/dev/null || true
mv run_demo_v2.sh OLD_VERSIONS/shell_scripts/ 2>/dev/null || true
mv run_complete_demo.sh OLD_VERSIONS/shell_scripts/ 2>/dev/null || true
mv run_complete_viz.sh OLD_VERSIONS/shell_scripts/ 2>/dev/null || true
mv run_real_ego_demo.sh OLD_VERSIONS/shell_scripts/ 2>/dev/null || true
mv generate_all_stages_data.py OLD_VERSIONS/scripts/ 2>/dev/null || true
mv generate_stage1_data.py OLD_VERSIONS/scripts/ 2>/dev/null || true

# 6. 归档旧的文档
mv ANIMATION_*.md OLD_VERSIONS/docs/ 2>/dev/null || true
mv BUGFIX_*.md OLD_VERSIONS/docs/ 2>/dev/null || true
mv DEMO_*.md OLD_VERSIONS/docs/ 2>/dev/null || true
mv ENHANCED_*.md OLD_VERSIONS/docs/ 2>/dev/null || true
mv IMPROVED_*.md OLD_VERSIONS/docs/ 2>/dev/null || true
mv LIVE_*.md OLD_VERSIONS/docs/ 2>/dev/null || true
mv REAL_DATA_*.md OLD_VERSIONS/docs/ 2>/dev/null || true
mv STATS_*.md OLD_VERSIONS/docs/ 2>/dev/null || true
mv TASK_*.md OLD_VERSIONS/docs/ 2>/dev/null || true
mv VISUALIZATION_*.md OLD_VERSIONS/docs/ 2>/dev/null || true
mv results/COLOR_*.md OLD_VERSIONS/docs/ 2>/dev/null || true
mv results/COMPLETION_*.md OLD_VERSIONS/docs/ 2>/dev/null || true
mv results/HIGHLIGHT_*.md OLD_VERSIONS/docs/ 2>/dev/null || true

echo "✅ 归档完成！"
echo ""
echo "📁 保留的核心文件："
echo "  - results/animated_attack_demo.html (最终演示)"
echo "  - results/real_data_demo.json (数据文件)"
echo "  - main_experiment_unified.py (实验代码)"
echo "  - generate_real_demo_data.py (数据生成)"
echo "  - visualize_interactive_dashboard.py (可视化)"
echo "  - attack/ defense/ models/ (核心算法)"
echo ""
echo "📦 归档的文件位于: OLD_VERSIONS/"
```

---

## 🎯 整理后的项目结构

```
Anonymous/
├── results/
│   ├── animated_attack_demo.html      ⭐ 最终演示页面
│   ├── real_data_demo.json            ⭐ 演示数据
│   ├── test_highlight.html            ⭐ 测试页面
│   ├── figures/                       # 实验结果图表
│   └── unified/                       # 实验结果数据
│
├── main_experiment_unified.py         ⭐ 统一实验入口
├── generate_real_demo_data.py         ⭐ 数据生成脚本
├── visualize_interactive_dashboard.py ⭐ 最新可视化（如果用到）
│
├── attack/                            ⭐ 攻击算法
├── defense/                           ⭐ 防御算法
├── models/                            ⭐ 模型代码
├── preprocessing/                     ⭐ 预处理
├── utils/                             ⭐ 工具函数
├── data/                              ⭐ 数据集
│
├── README.md                          # 项目说明
├── requirements.txt                   # 依赖列表
│
└── OLD_VERSIONS/                      📦 归档的旧版本
    ├── scripts/                       # 旧的Python脚本
    ├── html/                          # 旧的HTML页面
    ├── data/                          # 旧的数据文件
    ├── docs/                          # 旧的文档
    └── shell_scripts/                 # 旧的shell脚本
```

---

## 🚀 使用最新版本

### 1. 查看演示
```bash
cd results
python3 -m http.server 9000
# 访问: http://localhost:9000/animated_attack_demo.html
```

### 2. 运行实验
```bash
python3 main_experiment_unified.py --dataset facebook_ego --ego_id 0
```

### 3. 生成新数据
```bash
python3 generate_real_demo_data.py
```

---

**整理日期**: 2026-01-03
**保留版本**: Final v1.0

