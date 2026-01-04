"""
交互式动态仪表板生成器
展示三步骤攻击原理、防御策略和攻防对抗过程
包含动画、交互元素和数据可视化
"""

import json
import os
from datetime import datetime
from typing import Dict, List
import numpy as np


class InteractiveDashboardGenerator:
    """交互式仪表板生成器"""
    
    def __init__(self, results_data: Dict, output_path: str = 'results/figures/interactive_dashboard.html'):
        """
        初始化仪表板生成器
        
        Args:
            results_data: 实验结果数据
            output_path: 输出HTML路径
        """
        self.results = results_data
        self.output_path = output_path
        self.dataset_name = results_data.get('dataset', 'Unknown')
    
    def generate(self):
        """生成完整的交互式仪表板"""
        html_content = self._generate_html_structure()
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ 交互式仪表板已生成: {self.output_path}")
    
    def _generate_html_structure(self) -> str:
        """生成完整的HTML结构"""
        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>社交网络隐私保护 - 交互式仪表板</title>
    {self._generate_styles()}
</head>
<body>
    {self._generate_header()}
    {self._generate_navigation()}
    
    <div class="container">
        {self._generate_overview_section()}
        {self._generate_three_steps_animation()}
        {self._generate_attack_defense_battle()}
        {self._generate_methods_comparison()}
        {self._generate_detailed_results()}
    </div>
    
    {self._generate_scripts()}
</body>
</html>"""
    
    def _generate_styles(self) -> str:
        """生成CSS样式"""
        return """
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        :root {
            --primary-color: #667eea;
            --secondary-color: #764ba2;
            --success-color: #06A77D;
            --danger-color: #C73E1D;
            --warning-color: #F18F01;
            --bg-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--bg-gradient);
            min-height: 100vh;
            padding-bottom: 50px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        /* Header Styles */
        .header {
            background: white;
            padding: 40px;
            text-align: center;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            border-radius: 15px;
            margin-bottom: 30px;
            animation: slideDown 0.6s ease-out;
        }
        
        .header h1 {
            font-size: 3em;
            background: var(--bg-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 15px;
        }
        
        .header .subtitle {
            font-size: 1.3em;
            color: #666;
            margin-bottom: 10px;
        }
        
        .header .dataset-info {
            display: inline-block;
            padding: 10px 25px;
            background: var(--bg-gradient);
            color: white;
            border-radius: 25px;
            font-weight: bold;
            margin-top: 15px;
        }
        
        /* Navigation */
        .nav-tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
            flex-wrap: wrap;
            justify-content: center;
        }
        
        .nav-tab {
            padding: 15px 30px;
            background: white;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-size: 1.1em;
            font-weight: bold;
            color: var(--primary-color);
            transition: all 0.3s ease;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .nav-tab:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
        
        .nav-tab.active {
            background: var(--bg-gradient);
            color: white;
        }
        
        /* Section Styles */
        .section {
            background: white;
            border-radius: 15px;
            padding: 40px;
            margin-bottom: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            display: none;
            animation: fadeIn 0.5s ease-out;
        }
        
        .section.active {
            display: block;
        }
        
        .section-title {
            font-size: 2.5em;
            color: var(--secondary-color);
            margin-bottom: 30px;
            text-align: center;
            position: relative;
            padding-bottom: 15px;
        }
        
        .section-title::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 100px;
            height: 4px;
            background: var(--bg-gradient);
            border-radius: 2px;
        }
        
        /* Overview Cards */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .stat-card {
            padding: 30px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 15px;
            text-align: center;
            transition: transform 0.3s ease;
            animation: popIn 0.6s ease-out backwards;
        }
        
        .stat-card:nth-child(1) { animation-delay: 0.1s; }
        .stat-card:nth-child(2) { animation-delay: 0.2s; }
        .stat-card:nth-child(3) { animation-delay: 0.3s; }
        .stat-card:nth-child(4) { animation-delay: 0.4s; }
        
        .stat-card:hover {
            transform: translateY(-10px) scale(1.05);
        }
        
        .stat-card h3 {
            color: var(--secondary-color);
            font-size: 1.2em;
            margin-bottom: 15px;
        }
        
        .stat-value {
            font-size: 3em;
            font-weight: bold;
            background: var(--bg-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 10px 0;
        }
        
        .stat-label {
            color: #666;
            font-size: 0.9em;
        }
        
        /* Three Steps Animation */
        .steps-container {
            display: flex;
            justify-content: space-around;
            margin: 50px 0;
            flex-wrap: wrap;
            gap: 30px;
        }
        
        .step-card {
            flex: 1;
            min-width: 280px;
            padding: 30px;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            position: relative;
            overflow: hidden;
            transition: all 0.5s ease;
            animation: slideUp 0.8s ease-out backwards;
        }
        
        .step-card:nth-child(1) {
            animation-delay: 0.2s;
            border-top: 5px solid #667eea;
        }
        .step-card:nth-child(2) {
            animation-delay: 0.4s;
            border-top: 5px solid #F18F01;
        }
        .step-card:nth-child(3) {
            animation-delay: 0.6s;
            border-top: 5px solid #06A77D;
        }
        
        .step-card:hover {
            transform: translateY(-15px);
            box-shadow: 0 20px 50px rgba(0,0,0,0.2);
        }
        
        .step-number {
            position: absolute;
            top: 10px;
            right: 10px;
            width: 50px;
            height: 50px;
            background: var(--bg-gradient);
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5em;
            font-weight: bold;
        }
        
        .step-icon {
            font-size: 4em;
            text-align: center;
            margin-bottom: 20px;
        }
        
        .step-title {
            font-size: 1.5em;
            color: var(--secondary-color);
            margin-bottom: 15px;
            text-align: center;
        }
        
        .step-description {
            color: #666;
            line-height: 1.6;
            text-align: center;
        }
        
        .step-animation {
            margin-top: 20px;
            height: 150px;
            position: relative;
            border-top: 2px dashed #ddd;
            padding-top: 20px;
        }
        
        /* Battle Animation */
        .battle-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 40px;
            margin: 50px 0;
            flex-wrap: wrap;
        }
        
        .battle-side {
            flex: 1;
            min-width: 300px;
            padding: 40px;
            border-radius: 15px;
            text-align: center;
            position: relative;
        }
        
        .battle-side.attack {
            background: linear-gradient(135deg, #C73E1D 0%, #F18F01 100%);
            color: white;
            animation: pulse 2s infinite;
        }
        
        .battle-side.defense {
            background: linear-gradient(135deg, #06A77D 0%, #45B7D1 100%);
            color: white;
            animation: pulse 2s infinite 1s;
        }
        
        .battle-icon {
            font-size: 5em;
            margin-bottom: 20px;
        }
        
        .battle-title {
            font-size: 2em;
            margin-bottom: 15px;
            font-weight: bold;
        }
        
        .battle-score {
            font-size: 3em;
            font-weight: bold;
            margin: 20px 0;
        }
        
        .battle-middle {
            flex: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 20px;
        }
        
        .battle-vs {
            font-size: 3em;
            font-weight: bold;
            color: white;
            background: var(--secondary-color);
            width: 80px;
            height: 80px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            animation: rotate 4s linear infinite;
        }
        
        .battle-arrow {
            font-size: 3em;
            color: white;
            animation: bounce 1s infinite alternate;
        }
        
        /* Methods Comparison */
        .methods-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 25px;
            margin: 40px 0;
        }
        
        .method-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #e0e7ff 100%);
            border-radius: 15px;
            padding: 30px;
            position: relative;
            overflow: hidden;
            transition: all 0.4s ease;
        }
        
        .method-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 5px;
            background: var(--bg-gradient);
        }
        
        .method-card:hover {
            transform: translateY(-10px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.2);
        }
        
        .method-name {
            font-size: 1.4em;
            color: var(--secondary-color);
            margin-bottom: 20px;
            font-weight: bold;
        }
        
        .method-metrics {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .metric-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .metric-label {
            color: #666;
            font-size: 0.95em;
        }
        
        .metric-value {
            font-size: 1.3em;
            font-weight: bold;
            color: var(--primary-color);
        }
        
        .metric-bar {
            height: 10px;
            background: #e0e0e0;
            border-radius: 5px;
            overflow: hidden;
            margin-top: 5px;
        }
        
        .metric-fill {
            height: 100%;
            background: var(--bg-gradient);
            border-radius: 5px;
            transition: width 1s ease-out;
        }
        
        /* Charts Container */
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
            margin: 40px 0;
        }
        
        .chart-card {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }
        
        .chart-title {
            font-size: 1.5em;
            color: var(--secondary-color);
            margin-bottom: 20px;
            text-align: center;
            font-weight: bold;
        }
        
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        @keyframes slideDown {
            from {
                opacity: 0;
                transform: translateY(-50px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(50px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes popIn {
            0% {
                opacity: 0;
                transform: scale(0.5);
            }
            70% {
                transform: scale(1.1);
            }
            100% {
                opacity: 1;
                transform: scale(1);
            }
        }
        
        @keyframes pulse {
            0%, 100% {
                transform: scale(1);
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            50% {
                transform: scale(1.05);
                box-shadow: 0 15px 40px rgba(0,0,0,0.3);
            }
        }
        
        @keyframes bounce {
            from { transform: translateY(0); }
            to { transform: translateY(-10px); }
        }
        
        @keyframes rotate {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        
        /* Node Animation in Steps */
        .node {
            position: absolute;
            width: 15px;
            height: 15px;
            background: var(--primary-color);
            border-radius: 50%;
            animation: float 3s ease-in-out infinite;
        }
        
        .node:nth-child(1) { left: 10%; top: 30%; animation-delay: 0s; }
        .node:nth-child(2) { left: 30%; top: 60%; animation-delay: 0.5s; }
        .node:nth-child(3) { left: 50%; top: 20%; animation-delay: 1s; }
        .node:nth-child(4) { left: 70%; top: 70%; animation-delay: 1.5s; }
        .node:nth-child(5) { left: 90%; top: 40%; animation-delay: 2s; }
        
        @keyframes float {
            0%, 100% { transform: translateY(0) scale(1); opacity: 0.6; }
            50% { transform: translateY(-20px) scale(1.2); opacity: 1; }
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .header h1 {
                font-size: 2em;
            }
            
            .steps-container,
            .battle-container {
                flex-direction: column;
            }
            
            .charts-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
"""
    
    def _generate_header(self) -> str:
        """生成页面头部"""
        stats = self.results.get('graph_stats', {})
        return f"""
    <div class="header">
        <h1>🔐 社交网络隐私保护</h1>
        <div class="subtitle">从攻击到防御的完整研究</div>
        <div class="dataset-info">数据集: {self.dataset_name.upper()} | 节点: {stats.get('nodes', 'N/A')} | 边: {stats.get('edges', 'N/A')}</div>
        <p style="margin-top: 15px; color: #999;">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
"""
    
    def _generate_navigation(self) -> str:
        """生成导航标签"""
        return """
    <div class="nav-tabs">
        <button class="nav-tab active" onclick="showSection('overview')">📊 概览</button>
        <button class="nav-tab" onclick="showSection('three-steps')">🔍 三步骤原理</button>
        <button class="nav-tab" onclick="showSection('battle')">⚔️ 攻防对抗</button>
        <button class="nav-tab" onclick="showSection('methods')">🧪 方法对比</button>
        <button class="nav-tab" onclick="showSection('results')">📈 详细结果</button>
    </div>
"""
    
    def _generate_overview_section(self) -> str:
        """生成概览部分"""
        # 提取关键统计数据
        deanon_data = self.results.get('deanonymization', [])
        attr_data = self.results.get('attribute_inference', [])
        robust_data = self.results.get('robustness', [])
        defense_data = self.results.get('defense', [])
        
        # 计算最佳攻击准确率
        best_deanon_acc = max([d['accuracy'] for d in deanon_data], default=0) * 100
        
        # 计算最佳属性推断准确率
        best_attr_acc = max([a['accuracy'] for a in attr_data], default=0) * 100
        
        # 计算鲁棒性临界点
        critical_point = 50  # 默认值
        if robust_data:
            # 找到准确率下降到初始值50%的点
            accuracies = [(r['missing_ratio'], r['accuracy']) for r in robust_data]
            if accuracies:
                initial_acc = max(deanon_data, key=lambda x: x['accuracy'])['accuracy'] if deanon_data else 1.0
                for ratio, acc in sorted(accuracies):
                    if acc < initial_acc * 0.5:
                        critical_point = int((1 - ratio) * 100)
                        break
        
        # 计算最佳防御效果
        best_defense = "ε=1.0"
        if defense_data:
            # 寻找平衡点
            best_defense = f"ε={defense_data[0].get('epsilon', 1.0)}"
        
        return f"""
    <div class="section active" id="overview">
        <h2 class="section-title">实验结果概览</h2>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>🎯 身份去匿名化</h3>
                <div class="stat-value">{best_deanon_acc:.1f}%</div>
                <div class="stat-label">最佳攻击准确率</div>
            </div>
            
            <div class="stat-card">
                <h3>🔍 属性推断</h3>
                <div class="stat-value">{best_attr_acc:.1f}%</div>
                <div class="stat-label">最佳推断准确率</div>
            </div>
            
            <div class="stat-card">
                <h3>💪 鲁棒性临界点</h3>
                <div class="stat-value">{critical_point}%</div>
                <div class="stat-label">图完整度阈值</div>
            </div>
            
            <div class="stat-card">
                <h3>🛡️ 推荐防御参数</h3>
                <div class="stat-value">{best_defense}</div>
                <div class="stat-label">差分隐私预算</div>
            </div>
        </div>
    </div>
"""
    
    def _generate_three_steps_animation(self) -> str:
        """生成三步骤动画"""
        return """
    <div class="section" id="three-steps">
        <h2 class="section-title">🔍 三步骤攻击原理演示</h2>
        
        <div class="steps-container">
            <div class="step-card">
                <div class="step-number">1</div>
                <div class="step-icon">🎯</div>
                <div class="step-title">身份去匿名化</div>
                <div class="step-description">
                    通过结构指纹识别，将匿名图中的节点映射回原始身份。利用度序列、邻居结构等特征进行匹配。
                </div>
                <div class="step-animation">
                    <div class="node"></div>
                    <div class="node"></div>
                    <div class="node"></div>
                    <div class="node"></div>
                    <div class="node"></div>
                </div>
            </div>
            
            <div class="step-card">
                <div class="step-number">2</div>
                <div class="step-icon">🔎</div>
                <div class="step-title">属性推断攻击</div>
                <div class="step-description">
                    基于已知的部分节点属性和图结构，推断其他节点的敏感属性。利用标签传播和图神经网络。
                </div>
                <div class="step-animation">
                    <div class="node"></div>
                    <div class="node"></div>
                    <div class="node"></div>
                    <div class="node"></div>
                    <div class="node"></div>
                </div>
            </div>
            
            <div class="step-card">
                <div class="step-number">3</div>
                <div class="step-icon">🛡️</div>
                <div class="step-title">隐私防御机制</div>
                <div class="step-description">
                    采用差分隐私、K-匿名性、特征扰动等技术，在保持图效用的同时降低隐私泄露风险。
                </div>
                <div class="step-animation">
                    <div class="node"></div>
                    <div class="node"></div>
                    <div class="node"></div>
                    <div class="node"></div>
                    <div class="node"></div>
                </div>
            </div>
        </div>
        
        <div style="margin-top: 50px; padding: 30px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 15px;">
            <h3 style="text-align: center; color: var(--secondary-color); margin-bottom: 20px; font-size: 1.8em;">工作流程</h3>
            <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; gap: 20px;">
                <div style="text-align: center;">
                    <div style="width: 100px; height: 100px; background: var(--bg-gradient); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-size: 2em; margin: 0 auto 10px;">📊</div>
                    <div>原始社交网络</div>
                </div>
                <div style="font-size: 2em; color: var(--primary-color);">→</div>
                <div style="text-align: center;">
                    <div style="width: 100px; height: 100px; background: var(--bg-gradient); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-size: 2em; margin: 0 auto 10px;">🎭</div>
                    <div>匿名化处理</div>
                </div>
                <div style="font-size: 2em; color: var(--primary-color);">→</div>
                <div style="text-align: center;">
                    <div style="width: 100px; height: 100px; background: var(--bg-gradient); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-size: 2em; margin: 0 auto 10px;">⚔️</div>
                    <div>多维度攻击</div>
                </div>
                <div style="font-size: 2em; color: var(--primary-color);">→</div>
                <div style="text-align: center;">
                    <div style="width: 100px; height: 100px; background: var(--bg-gradient); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-size: 2em; margin: 0 auto 10px;">🛡️</div>
                    <div>防御评估</div>
                </div>
            </div>
        </div>
    </div>
"""
    
    def _generate_attack_defense_battle(self) -> str:
        """生成攻防对抗部分"""
        # 计算攻击成功率
        deanon_data = self.results.get('deanonymization', [])
        attack_score = max([d['accuracy'] for d in deanon_data], default=0) * 100 if deanon_data else 0
        
        # 计算防御有效性
        defense_data = self.results.get('defense', [])
        defense_score = 0
        if defense_data:
            # 防御得分 = 隐私增益 (假设差分隐私能降低攻击成功率)
            # 这里简化为边保留率
            defense_score = (1 - defense_data[0].get('edge_preservation', 0)) * 100
            if defense_score < 1:
                defense_score = defense_data[0].get('utility_score', 0) * 100
        
        return f"""
    <div class="section" id="battle">
        <h2 class="section-title">⚔️ 攻击 vs 防御对抗演示</h2>
        
        <div class="battle-container">
            <div class="battle-side attack">
                <div class="battle-icon">⚔️</div>
                <div class="battle-title">攻击方</div>
                <div class="battle-score">{attack_score:.1f}%</div>
                <div>最高攻击成功率</div>
                <div style="margin-top: 20px; font-size: 0.9em;">
                    <div>✓ 结构指纹识别</div>
                    <div>✓ 节点特征匹配</div>
                    <div>✓ 图对齐算法</div>
                </div>
            </div>
            
            <div class="battle-middle">
                <div class="battle-vs">VS</div>
                <div class="battle-arrow">⬇️</div>
                <div style="color: white; font-size: 1.2em; font-weight: bold;">动态博弈</div>
            </div>
            
            <div class="battle-side defense">
                <div class="battle-icon">🛡️</div>
                <div class="battle-title">防御方</div>
                <div class="battle-score">{defense_score:.1f}%</div>
                <div>隐私保护强度</div>
                <div style="margin-top: 20px; font-size: 0.9em;">
                    <div>✓ 差分隐私</div>
                    <div>✓ K-匿名性</div>
                    <div>✓ 特征扰动</div>
                </div>
            </div>
        </div>
        
        <div style="margin-top: 50px; text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;">
            <h3 style="font-size: 2em; margin-bottom: 20px;">💡 关键洞察</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 30px; margin-top: 30px;">
                <div style="background: rgba(255,255,255,0.1); padding: 25px; border-radius: 10px;">
                    <div style="font-size: 2.5em; margin-bottom: 10px;">🎯</div>
                    <div style="font-size: 1.3em; font-weight: bold;">攻击多样性</div>
                    <div style="margin-top: 10px; opacity: 0.9;">结构+特征组合攻击更有效</div>
                </div>
                <div style="background: rgba(255,255,255,0.1); padding: 25px; border-radius: 10px;">
                    <div style="font-size: 2.5em; margin-bottom: 10px;">⚖️</div>
                    <div style="font-size: 1.3em; font-weight: bold;">隐私-效用权衡</div>
                    <div style="margin-top: 10px; opacity: 0.9;">需要在保护和可用性间平衡</div>
                </div>
                <div style="background: rgba(255,255,255,0.1); padding: 25px; border-radius: 10px;">
                    <div style="font-size: 2.5em; margin-bottom: 10px;">🔄</div>
                    <div style="font-size: 1.3em; font-weight: bold;">动态防御策略</div>
                    <div style="margin-top: 10px; opacity: 0.9;">根据威胁调整防御强度</div>
                </div>
            </div>
        </div>
    </div>
"""
    
    def _generate_methods_comparison(self) -> str:
        """生成方法对比部分"""
        deanon_data = self.results.get('deanonymization', [])
        
        # 按方法分组并计算平均性能
        method_stats = {}
        for item in deanon_data:
            method = item['method']
            if method not in method_stats:
                method_stats[method] = []
            method_stats[method].append(item)
        
        method_cards_html = ""
        for method, items in method_stats.items():
            avg_acc = np.mean([i['accuracy'] for i in items]) * 100
            avg_prec5 = np.mean([i['precision@5'] for i in items]) * 100
            avg_mrr = np.mean([i['mrr'] for i in items]) * 100
            
            method_cards_html += f"""
            <div class="method-card">
                <div class="method-name">{method}</div>
                <div class="method-metrics">
                    <div>
                        <div class="metric-row">
                            <span class="metric-label">准确率</span>
                            <span class="metric-value">{avg_acc:.1f}%</span>
                        </div>
                        <div class="metric-bar">
                            <div class="metric-fill" style="width: {avg_acc}%"></div>
                        </div>
                    </div>
                    <div>
                        <div class="metric-row">
                            <span class="metric-label">Precision@5</span>
                            <span class="metric-value">{avg_prec5:.1f}%</span>
                        </div>
                        <div class="metric-bar">
                            <div class="metric-fill" style="width: {avg_prec5}%"></div>
                        </div>
                    </div>
                    <div>
                        <div class="metric-row">
                            <span class="metric-label">MRR</span>
                            <span class="metric-value">{avg_mrr:.1f}%</span>
                        </div>
                        <div class="metric-bar">
                            <div class="metric-fill" style="width: {avg_mrr}%"></div>
                        </div>
                    </div>
                </div>
            </div>
"""
        
        return f"""
    <div class="section" id="methods">
        <h2 class="section-title">🧪 攻击方法性能对比</h2>
        
        <div class="methods-grid">
            {method_cards_html}
        </div>
    </div>
"""
    
    def _generate_detailed_results(self) -> str:
        """生成详细结果部分（包含图表）"""
        return """
    <div class="section" id="results">
        <h2 class="section-title">📈 详细实验结果</h2>
        
        <div class="charts-grid">
            <div class="chart-card">
                <div class="chart-title">身份去匿名化结果</div>
                <canvas id="deanonChart"></canvas>
            </div>
            
            <div class="chart-card">
                <div class="chart-title">属性推断结果</div>
                <canvas id="attrChart"></canvas>
            </div>
            
            <div class="chart-card">
                <div class="chart-title">鲁棒性测试</div>
                <canvas id="robustChart"></canvas>
            </div>
            
            <div class="chart-card">
                <div class="chart-title">防御效果评估</div>
                <canvas id="defenseChart"></canvas>
            </div>
        </div>
    </div>
"""
    
    def _generate_scripts(self) -> str:
        """生成JavaScript脚本"""
        # 准备图表数据
        deanon_data = self.results.get('deanonymization', [])
        attr_data = self.results.get('attribute_inference', [])
        robust_data = self.results.get('robustness', [])
        defense_data = self.results.get('defense', [])
        
        # 处理去匿名化数据（只取温和级别）
        deanon_mild = [d for d in deanon_data if d.get('level') == '温和']
        deanon_labels = [d['method'] for d in deanon_mild]
        deanon_values = [d['accuracy'] * 100 for d in deanon_mild]
        
        # 处理属性推断数据
        attr_labels = [f"{a['method']}<br>({int(a['hide_ratio']*100)}% hidden)" for a in attr_data]
        attr_values = [a['accuracy'] * 100 for a in attr_data]
        
        # 处理鲁棒性数据
        robust_labels = [f"{int((1-r['missing_ratio'])*100)}%" for r in robust_data]
        robust_values = [r['accuracy'] * 100 for r in robust_data]
        
        # 处理防御数据
        defense_labels = [f"ε={d['epsilon']}" for d in defense_data]
        defense_values = [d.get('utility_score', 0) * 100 for d in defense_data]
        
        return f"""
    <script>
        // 导航切换功能
        function showSection(sectionId) {{
            // 隐藏所有部分
            document.querySelectorAll('.section').forEach(section => {{
                section.classList.remove('active');
            }});
            
            // 显示选中的部分
            document.getElementById(sectionId).classList.add('active');
            
            // 更新导航标签状态
            document.querySelectorAll('.nav-tab').forEach(tab => {{
                tab.classList.remove('active');
            }});
            event.target.classList.add('active');
        }}
        
        // 图表配置
        const chartColors = {{
            primary: 'rgba(102, 126, 234, 0.8)',
            secondary: 'rgba(118, 75, 162, 0.8)',
            success: 'rgba(6, 167, 125, 0.8)',
            danger: 'rgba(199, 62, 29, 0.8)',
            warning: 'rgba(241, 143, 1, 0.8)'
        }};
        
        // 创建去匿名化图表
        const deanonCtx = document.getElementById('deanonChart');
        if (deanonCtx) {{
            new Chart(deanonCtx, {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(deanon_labels)},
                    datasets: [{{
                        label: '准确率 (%)',
                        data: {json.dumps(deanon_values)},
                        backgroundColor: chartColors.primary,
                        borderColor: chartColors.primary.replace('0.8', '1'),
                        borderWidth: 2
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {{
                        legend: {{
                            display: true,
                            position: 'top'
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: 100,
                            title: {{
                                display: true,
                                text: '准确率 (%)'
                            }}
                        }}
                    }}
                }}
            }});
        }}
        
        // 创建属性推断图表
        const attrCtx = document.getElementById('attrChart');
        if (attrCtx) {{
            new Chart(attrCtx, {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(attr_labels)},
                    datasets: [{{
                        label: '准确率 (%)',
                        data: {json.dumps(attr_values)},
                        backgroundColor: chartColors.warning,
                        borderColor: chartColors.warning.replace('0.8', '1'),
                        borderWidth: 2
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {{
                        legend: {{
                            display: true
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: 100,
                            title: {{
                                display: true,
                                text: '准确率 (%)'
                            }}
                        }}
                    }}
                }}
            }});
        }}
        
        // 创建鲁棒性图表
        const robustCtx = document.getElementById('robustChart');
        if (robustCtx) {{
            new Chart(robustCtx, {{
                type: 'line',
                data: {{
                    labels: {json.dumps(robust_labels)},
                    datasets: [{{
                        label: '攻击准确率 (%)',
                        data: {json.dumps(robust_values)},
                        backgroundColor: 'rgba(102, 126, 234, 0.2)',
                        borderColor: chartColors.primary,
                        borderWidth: 3,
                        pointRadius: 6,
                        pointHoverRadius: 8,
                        tension: 0.4
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {{
                        legend: {{
                            display: true
                        }}
                    }},
                    scales: {{
                        x: {{
                            title: {{
                                display: true,
                                text: '图完整度 (%)'
                            }}
                        }},
                        y: {{
                            beginAtZero: true,
                            title: {{
                                display: true,
                                text: '准确率 (%)'
                            }}
                        }}
                    }}
                }}
            }});
        }}
        
        // 创建防御效果图表
        const defenseCtx = document.getElementById('defenseChart');
        if (defenseCtx) {{
            new Chart(defenseCtx, {{
                type: 'line',
                data: {{
                    labels: {json.dumps(defense_labels)},
                    datasets: [{{
                        label: '效用保持 (%)',
                        data: {json.dumps(defense_values)},
                        backgroundColor: 'rgba(6, 167, 125, 0.2)',
                        borderColor: chartColors.success,
                        borderWidth: 3,
                        pointRadius: 6,
                        pointHoverRadius: 8,
                        tension: 0.4
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {{
                        legend: {{
                            display: true
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: 100,
                            title: {{
                                display: true,
                                text: '效用 (%)'
                            }}
                        }}
                    }}
                }}
            }});
        }}
        
        // 页面加载完成后的动画
        window.addEventListener('load', function() {{
            // 触发进度条动画
            setTimeout(() => {{
                document.querySelectorAll('.metric-fill').forEach(bar => {{
                    bar.style.width = bar.style.width;
                }});
            }}, 500);
        }});
    </script>
"""


def create_interactive_dashboard(json_path: str, output_path: str = None):
    """
    创建交互式仪表板
    
    Args:
        json_path: JSON结果文件路径
        output_path: 输出HTML路径
    """
    # 读取JSON数据
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    # 确定输出路径
    if output_path is None:
        dataset_name = results.get('dataset', 'unknown')
        ego_id = results.get('ego_id')
        if ego_id is not None:
            output_path = f'results/figures/{dataset_name}_ego{ego_id}_interactive.html'
        else:
            output_path = f'results/figures/{dataset_name}_interactive.html'
    
    # 生成仪表板
    generator = InteractiveDashboardGenerator(results, output_path)
    generator.generate()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python visualize_interactive_dashboard.py <json_path> [output_path]")
        sys.exit(1)
    
    json_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    create_interactive_dashboard(json_path, output_path)
















