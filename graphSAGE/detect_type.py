import pickle
import sys
from pathlib import Path


def analyze_pickle():
    # --- 1. 强壮的路径计算逻辑 ---
    # 获取当前脚本 (detect_type.py) 的绝对路径
    current_script_path = Path(__file__).resolve()

    # 获取当前脚本所在的目录 (即 .../anon/graphSAGE)
    script_dir = current_script_path.parent

    # 获取项目根目录 (即 .../anon) - 向上一级
    project_root = script_dir.parent

    # 拼接数据目录
    data_dir = project_root / "data"

    # 定义具体文件路径 (完全匹配你的截图结构)
    files = {
        "ORIG": data_dir / "processed" / "graph.gpickle",
        "ANON": data_dir / "anonymized" / "anonymous_graph.gpickle",
        "GT": data_dir / "anonymized" / "ground_truth.pkl"
    }

    print("=" * 60)
    print("🔍 路径诊断与数据结构分析")
    print("=" * 60)
    print(f"📍 脚本位置: {current_script_path}")
    print(f"📂 项目根目录: {project_root}")
    print(f"📂 数据目录: {data_dir}")
    print("-" * 60)

    # --- 2. 逐个分析文件 ---

    # [1] 分析原始图
    print(f"\n[1] 分析原始图: processed/graph.gpickle")
    print(f"    -> 完整路径: {files['ORIG']}")
    if not files['ORIG'].exists():
        print("❌ 文件不存在! 请检查 'data/processed' 目录下是否有该文件。")
    else:
        try:
            with open(files['ORIG'], 'rb') as f:
                G = pickle.load(f)
            # 获取第一个节点看看类型
            if hasattr(G, 'nodes'):
                nodes_list = list(G.nodes())
                if nodes_list:
                    node_sample = nodes_list[0]
                    print(f"    ✅ 加载成功!")
                    print(f"    - 图类型: {type(G)}")
                    print(f"    - 节点数: {len(nodes_list)}")
                    print(f"    - 节点ID类型: {type(node_sample)} (示例: {node_sample!r})")
                else:
                    print("    ⚠️ 图加载成功，但没有节点。")
            else:
                print(f"    ⚠️ 加载的对象不是 NetworkX 图: {type(G)}")
        except Exception as e:
            print(f"    ❌ 读取出错: {e}")

    # [2] 分析匿名图
    print(f"\n[2] 分析匿名图: anonymized/anonymous_graph.gpickle")
    print(f"    -> 完整路径: {files['ANON']}")
    if not files['ANON'].exists():
        print("❌ 文件不存在! 请检查 'data/anonymized' 目录下是否有该文件。")
    else:
        try:
            with open(files['ANON'], 'rb') as f:
                G = pickle.load(f)
            if hasattr(G, 'nodes'):
                nodes_list = list(G.nodes())
                if nodes_list:
                    node_sample = nodes_list[0]
                    print(f"    ✅ 加载成功!")
                    print(f"    - 节点ID类型: {type(node_sample)} (示例: {node_sample!r})")
                else:
                    print("    ⚠️ 图为空。")
        except Exception as e:
            print(f"    ❌ 读取出错: {e}")

    # [3] 分析 Ground Truth (最关键的部分)
    print(f"\n[3] 分析 Ground Truth: anonymized/ground_truth.pkl")
    print(f"    -> 完整路径: {files['GT']}")
    if not files['GT'].exists():
        print("❌ 文件不存在! 请检查 'data/anonymized' 目录下是否有该文件。")
    else:
        try:
            with open(files['GT'], 'rb') as f:
                data = pickle.load(f)

            print(f"    ✅ 加载成功! 顶层类型: {type(data)}")

            final_gt = data
            is_nested = False

            # 检查是否有 'node_mapping' 键
            if isinstance(data, dict) and 'node_mapping' in data:
                print("    ⚠️ 检测到嵌套结构: 包含键 'node_mapping'")
                final_gt = data['node_mapping']
                is_nested = True
            else:
                print("    ✅ 无嵌套结构 (是扁平字典)")

            # 检查具体的 Key-Value 类型
            if isinstance(final_gt, dict) and len(final_gt) > 0:
                k, v = list(final_gt.items())[0]
                print(f"    - 映射示例: {k!r} -> {v!r}")
                print(f"    - Key (原ID) 类型: {type(k)}")
                print(f"    - Val (匿ID) 类型: {type(v)}")

                print(f"\n💡 [修改建议]")
                if is_nested:
                    print(f"   Dataloader 代码应写为: ground_truth = pickle.load(f)['node_mapping']")
                else:
                    print(f"   Dataloader 代码应写为: ground_truth = pickle.load(f)")

                print(f"   ID类型处理: 确保 dataloader 中转换 logic 匹配上述类型 ({type(k).__name__})")
            else:
                print("    ⚠️ Ground Truth 字典为空")

        except Exception as e:
            print(f"    ❌ 读取出错: {e}")


if __name__ == "__main__":
    analyze_pickle()