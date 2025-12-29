"""
HTML Visualization Script - Generate Interactive Charts with Chart.js
No matplotlib dependency, directly generates HTML files
"""

import json
import os
from pathlib import Path
from datetime import datetime


class HTMLVisualizer:
    """HTML Visualizer"""
    
    def __init__(self, results_file=None):
        """Initialize"""
        if results_file is None:
            results_file = self._find_latest_results()
        
        self.results_file = results_file
        
        # Extract dataset name from filename (format: {dataset}_{timestamp}_results.json)
        filename = Path(results_file).stem  # Remove .json extension
        parts = filename.split('_')
        if len(parts) >= 2:
            # Assume format: dataset_timestamp_results, take first part
            self.dataset_name = parts[0]
        else:
            self.dataset_name = 'unknown'
        
        self.output_dir = Path('results/figures')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        print(f"✓ Loaded results file: {results_file}")
        print(f"✓ Detected dataset: {self.dataset_name}")
    
    def _find_latest_results(self):
        """Find the latest results file"""
        results_dir = Path('results/structural_fingerprint')
        json_files = list(results_dir.glob('*_results.json'))
        
        if not json_files:
            raise FileNotFoundError("No results file found! Please run experiment first.")
        
        latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
        return str(latest_file)
    
    def generate_html_dashboard(self):
        """Generate HTML Dashboard"""
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Structural Fingerprint - Experiment Results</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header p {{
            color: #666;
            font-size: 1.1em;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        
        .stat-card h3 {{
            color: #764ba2;
            font-size: 1.2em;
            margin-bottom: 15px;
        }}
        
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }}
        
        .stat-label {{
            color: #888;
            font-size: 0.9em;
        }}
        
        .chart-container {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        
        .chart-title {{
            color: #764ba2;
            font-size: 1.5em;
            margin-bottom: 20px;
            text-align: center;
            font-weight: bold;
        }}
        
        canvas {{
            max-height: 400px;
        }}
        
        .grid-2 {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
        }}
        
        .footer {{
            text-align: center;
            color: white;
            margin-top: 30px;
            padding: 20px;
            font-size: 0.9em;
        }}
        
        .badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
            margin: 5px;
        }}
        
        .badge-success {{
            background: #06A77D;
            color: white;
        }}
        
        .badge-warning {{
            background: #F18F01;
            color: white;
        }}
        
        .badge-danger {{
            background: #C73E1D;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 Structural Fingerprints - Experiment Results</h1>
            <p>Social Network Privacy: Attacks & Defense - Comprehensive Analysis Dashboard</p>
            <p style="margin-top: 10px; color: #999;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        {self._generate_stats_cards()}
        
        <div class="grid-2">
            {self._generate_identity_chart()}
            {self._generate_attribute_chart()}
        </div>
        
        {self._generate_robustness_chart()}
        
        <div class="grid-2">
            {self._generate_privacy_chart()}
            {self._generate_utility_chart()}
        </div>
        
        <div class="footer">
            <p>📊 Structural Fingerprints in Social Networks</p>
            <p style="margin-top: 10px;">From Multi-dimensional Attacks to DP-based Defense</p>
        </div>
    </div>
</body>
</html>
"""
        
        output_path = self.output_dir / f'{self.dataset_name}_dashboard.html'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\n✓ HTML dashboard generated: {output_path}")
        print(f"  Open in browser: file://{output_path.absolute()}")
        
        return output_path
    
    def _generate_stats_cards(self):
        """Generate statistics cards"""
        cards = []
        
        # Identity De-anonymization Best Result
        if 'stage1_identity' in self.results:
            best_acc = max(m['accuracy'] for m in self.results['stage1_identity'].values())
            best_method = [k for k, v in self.results['stage1_identity'].items() 
                          if v['accuracy'] == best_acc][0]
            
            cards.append(f"""
            <div class="stat-card">
                <h3>Identity De-anonymization</h3>
                <div class="stat-value">{best_acc*100:.2f}%</div>
                <div class="stat-label">{best_method}</div>
                <span class="badge badge-warning">Attack Accuracy</span>
            </div>
            """)
        
        # Attribute Inference Best Result
        if 'stage1_attribute' in self.results:
            best_acc = max(m['accuracy'] for m in self.results['stage1_attribute'].values())
            best_method = [k for k, v in self.results['stage1_attribute'].items() 
                          if v['accuracy'] == best_acc][0]
            
            cards.append(f"""
            <div class="stat-card">
                <h3>Attribute Inference</h3>
                <div class="stat-value">{best_acc*100:.1f}%</div>
                <div class="stat-label">{best_method}</div>
                <span class="badge badge-danger">Privacy Leakage</span>
            </div>
            """)
        
        # Critical Point
        if 'stage2_robustness' in self.results:
            data = self.results['stage2_robustness']
            completeness = sorted([float(k) for k in data.keys()], reverse=True)
            accuracies = [data[str(c)]['accuracy'] for c in completeness]
            baseline = accuracies[0]
            
            critical_point = None
            for i, (c, acc) in enumerate(zip(completeness, accuracies)):
                if (baseline - acc) / baseline >= 0.5:
                    critical_point = c * 100
                    break
            
            if critical_point:
                cards.append(f"""
                <div class="stat-card">
                    <h3>Critical Point</h3>
                    <div class="stat-value">{critical_point:.0f}%</div>
                    <div class="stat-label">Graph Completeness</div>
                    <span class="badge badge-warning">Robustness</span>
                </div>
                """)
        
        # Best Privacy Budget
        if 'stage3_defense' in self.results:
            best_epsilon = None
            best_gain = -999
            
            for eps, metrics in self.results['stage3_defense'].items():
                gain = metrics['privacy']['relative_privacy_gain']
                if gain > best_gain and gain >= 0.4:
                    best_gain = gain
                    best_epsilon = float(eps)
            
            if best_epsilon:
                cards.append(f"""
                <div class="stat-card">
                    <h3>Recommended ε</h3>
                    <div class="stat-value">ε = {best_epsilon}</div>
                    <div class="stat-label">Privacy Gain {best_gain*100:.1f}%</div>
                    <span class="badge badge-success">Differential Privacy</span>
                </div>
                """)
        
        return f'<div class="stats-grid">{"".join(cards)}</div>'
    
    def _generate_identity_chart(self):
        """Generate identity de-anonymization chart"""
        if 'stage1_identity' not in self.results:
            return ""
        
        data = self.results['stage1_identity']
        methods = list(data.keys())
        accuracies = [data[m]['accuracy'] * 100 for m in methods]
        improvements = [data[m]['improvement_factor'] for m in methods]
        
        return f"""
        <div class="chart-container">
            <div class="chart-title">Identity De-anonymization Attack Performance</div>
            <canvas id="identityChart"></canvas>
        </div>
        <script>
        new Chart(document.getElementById('identityChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(methods)},
                datasets: [{{
                    label: 'Accuracy (%)',
                    data: {json.dumps(accuracies)},
                    backgroundColor: 'rgba(46, 134, 171, 0.8)',
                    borderColor: 'rgba(46, 134, 171, 1)',
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
                    }},
                    tooltip: {{
                        callbacks: {{
                            afterLabel: function(context) {{
                                let idx = context.dataIndex;
                                let improvements = {json.dumps(improvements)};
                                return 'Improvement: ' + improvements[idx].toFixed(1) + 'x';
                            }}
                        }}
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Accuracy (%)'
                        }}
                    }}
                }}
            }}
        }});
        </script>
        """
    
    def _generate_attribute_chart(self):
        """Generate attribute inference chart"""
        if 'stage1_attribute' not in self.results:
            return ""
        
        data = self.results['stage1_attribute']
        methods = list(data.keys())
        accuracies = [data[m]['accuracy'] * 100 for m in methods]
        
        return f"""
        <div class="chart-container">
            <div class="chart-title">Attribute Inference Attack Performance</div>
            <canvas id="attributeChart"></canvas>
        </div>
        <script>
        new Chart(document.getElementById('attributeChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(methods)},
                datasets: [{{
                    label: 'Accuracy (%)',
                    data: {json.dumps(accuracies)},
                    backgroundColor: ['rgba(162, 59, 114, 0.8)', 'rgba(241, 143, 1, 0.8)'],
                    borderColor: ['rgba(162, 59, 114, 1)', 'rgba(241, 143, 1, 1)'],
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
                            text: 'Accuracy (%)'
                        }}
                    }}
                }}
            }}
        }});
        </script>
        """
    
    def _generate_robustness_chart(self):
        """Generate robustness curve"""
        if 'stage2_robustness' not in self.results:
            return ""
        
        data = self.results['stage2_robustness']
        completeness = sorted([float(k) for k in data.keys()], reverse=True)
        accuracies = [data[str(c)]['accuracy'] * 100 for c in completeness]
        completeness_pct = [c * 100 for c in completeness]
        
        return f"""
        <div class="chart-container">
            <div class="chart-title">Robustness Test - Attack Success Rate vs Graph Completeness</div>
            <canvas id="robustnessChart"></canvas>
        </div>
        <script>
        new Chart(document.getElementById('robustnessChart'), {{
            type: 'line',
            data: {{
                labels: {json.dumps(completeness_pct)},
                datasets: [{{
                    label: 'Attack Accuracy (%)',
                    data: {json.dumps(accuracies)},
                    backgroundColor: 'rgba(46, 134, 171, 0.2)',
                    borderColor: 'rgba(46, 134, 171, 1)',
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
                        display: true,
                        position: 'top'
                    }}
                }},
                scales: {{
                    x: {{
                        title: {{
                            display: true,
                            text: 'Graph Completeness (%)'
                        }}
                    }},
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Attack Accuracy (%)'
                        }}
                    }}
                }}
            }}
        }});
        </script>
        """
    
    def _generate_privacy_chart(self):
        """Generate privacy gain chart"""
        if 'stage3_defense' not in self.results:
            return ""
        
        data = self.results['stage3_defense']
        epsilons = sorted([float(k) for k in data.keys()])
        privacy_gains = [data[str(e)]['privacy']['relative_privacy_gain'] * 100 for e in epsilons]
        
        return f"""
        <div class="chart-container">
            <div class="chart-title">Differential Privacy Defense - Privacy Gain</div>
            <canvas id="privacyChart"></canvas>
        </div>
        <script>
        new Chart(document.getElementById('privacyChart'), {{
            type: 'line',
            data: {{
                labels: {json.dumps(['ε='+str(e) for e in epsilons])},
                datasets: [{{
                    label: 'Privacy Gain (%)',
                    data: {json.dumps(privacy_gains)},
                    backgroundColor: 'rgba(6, 167, 125, 0.2)',
                    borderColor: 'rgba(6, 167, 125, 1)',
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
                        title: {{
                            display: true,
                            text: 'Privacy Gain (%)'
                        }}
                    }}
                }}
            }}
        }});
        </script>
        """
    
    def _generate_utility_chart(self):
        """Generate utility preservation chart"""
        if 'stage3_defense' not in self.results:
            return ""
        
        data = self.results['stage3_defense']
        epsilons = sorted([float(k) for k in data.keys()])
        modularity = [data[str(e)]['utility'].get('modularity_preservation', 0) * 100 for e in epsilons]
        centrality = [data[str(e)]['utility'].get('centrality_preservation', 0) * 100 for e in epsilons]
        
        return f"""
        <div class="chart-container">
            <div class="chart-title">Differential Privacy Defense - Utility Preservation</div>
            <canvas id="utilityChart"></canvas>
        </div>
        <script>
        new Chart(document.getElementById('utilityChart'), {{
            type: 'line',
            data: {{
                labels: {json.dumps(['ε='+str(e) for e in epsilons])},
                datasets: [
                    {{
                        label: 'Modularity Preservation (%)',
                        data: {json.dumps(modularity)},
                        backgroundColor: 'rgba(46, 134, 171, 0.2)',
                        borderColor: 'rgba(46, 134, 171, 1)',
                        borderWidth: 3,
                        pointRadius: 6,
                        tension: 0.4
                    }},
                    {{
                        label: 'Centrality Preservation (%)',
                        data: {json.dumps(centrality)},
                        backgroundColor: 'rgba(162, 59, 114, 0.2)',
                        borderColor: 'rgba(162, 59, 114, 1)',
                        borderWidth: 3,
                        pointRadius: 6,
                        tension: 0.4
                    }}
                ]
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
                            text: 'Utility Preservation (%)'
                        }}
                    }}
                }}
            }}
        }});
        </script>
        """


def main():
    """Main function"""
    print("\n" + "="*70)
    print("Structural Fingerprint Project - HTML Visualization Script")
    print("="*70)
    
    try:
        visualizer = HTMLVisualizer()
        output_path = visualizer.generate_html_dashboard()
        
        print("\n✅ HTML visualization completed!")
        print(f"\n📊 Open in browser:")
        print(f"   file://{output_path.absolute()}")
        print(f"\nOr directly open: {output_path}")
        
        return 0
    except Exception as e:
        print(f"\n❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())

