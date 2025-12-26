"""
微博爬虫 - 调试版本
增加详细的错误信息和测试功能
"""

import requests
import json
import time
from pathlib import Path

# 测试cookies是否有效
def test_cookies(cookies):
    """测试cookies是否有效"""
    print("🔍 测试1: 检查cookies格式...")
    if not cookies or cookies == "YOUR_WEIBO_COOKIES_HERE":
        print("❌ 错误: cookies是占位符，需要替换为真实的cookies")
        return False
    
    print(f"✅ cookies长度: {len(cookies)} 字符")
    
    # 测试请求
    print("\n🔍 测试2: 测试API连接...")
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15',
        'Referer': 'https://m.weibo.cn/',
        'Cookie': cookies
    })
    
    # 测试简单的API
    test_url = "https://m.weibo.cn/api/config"
    try:
        response = session.get(test_url, timeout=10)
        print(f"   状态码: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"   ✅ API响应正常")
                print(f"   响应数据示例: {str(data)[:100]}...")
                return True
            except:
                print(f"   ⚠️  返回了非JSON数据")
                print(f"   响应内容: {response.text[:200]}...")
                return False
        else:
            print(f"   ❌ HTTP状态码异常: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ 请求失败: {e}")
        return False


def test_user_api(cookies, uid):
    """测试用户信息API"""
    print(f"\n🔍 测试3: 获取用户 {uid} 的信息...")
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15',
        'Referer': 'https://m.weibo.cn/',
        'Cookie': cookies
    })
    
    # 尝试多个API端点
    apis = [
        f"https://m.weibo.cn/api/container/getIndex?type=uid&value={uid}&containerid=100505{uid}",
        f"https://m.weibo.cn/profile/info?uid={uid}",
        f"https://weibo.com/ajax/profile/info?uid={uid}",
    ]
    
    for i, url in enumerate(apis, 1):
        print(f"\n   尝试API #{i}: {url}")
        try:
            response = session.get(url, timeout=10)
            print(f"   状态码: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"   ✅ 返回JSON数据")
                    print(f"   数据结构: {json.dumps(data, indent=2, ensure_ascii=False)[:300]}...")
                    
                    # 检查是否有错误信息
                    if 'ok' in data and data['ok'] == 0:
                        print(f"   ⚠️  API返回错误: {data.get('msg', 'Unknown error')}")
                    elif 'data' in data:
                        print(f"   ✅ API #{i} 可用！")
                        return True, url
                    
                except json.JSONDecodeError as e:
                    print(f"   ❌ JSON解析失败: {e}")
                    print(f"   响应内容: {response.text[:200]}...")
            else:
                print(f"   ❌ HTTP错误: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ 请求失败: {e}")
    
    return False, None


def get_fresh_cookies_guide():
    """显示获取cookies的详细指南"""
    print("\n" + "="*60)
    print("📖 如何获取有效的微博Cookies")
    print("="*60)
    print("""
方法1: 使用移动版微博 (m.weibo.cn) - 推荐

1. 打开浏览器（Chrome/Firefox）
2. 访问 https://m.weibo.cn
3. 登录你的微博账号
4. 按F12打开开发者工具
5. 切换到 Network (网络) 标签
6. 刷新页面 (F5)
7. 找到任意请求（如 config 或 getIndex）
8. 点击该请求，在右侧找到 Request Headers
9. 复制 Cookie 字段的完整值

Cookie示例（很长的字符串）:
SINAGLOBAL=xxx; ULV=xxx; XSRF-TOKEN=xxx; SCF=xxx; SUB=xxx; SUBP=xxx; ALF=xxx; WBPSESS=xxx

重要提示:
- Cookie会过期（通常几小时到几天）
- 不要分享你的Cookie（包含登录凭证）
- 确保复制完整，不要有换行或空格

方法2: 使用PC版微博 (weibo.com) - 备选

1. 访问 https://weibo.com
2. 登录账号
3. F12 → Network
4. 刷新页面
5. 找到任意XHR请求
6. 复制Cookie
    """)
    print("="*60)


def main():
    print("="*60)
    print("微博爬虫调试工具")
    print("="*60)
    
    # 从原始爬虫文件中读取cookies
    cookies = "SINAGLOBAL=6740185828856.008.1764257392979; ULV=1764257392981:1:1:1:6740185828856.008.1764257392979:; XSRF-TOKEN=Myn4TmTnG35cSjgyYPIJfvmV; SCF=AjiMSHwPp3pk5eVrMx10d6WYKiUi8q5VEC2hifoXmNfxm-mQDE2IPwP4DaI7i_6W3iyQ4sat5D1N02_MdRCywNM.; SUB=_2A25EStb9DeRhGeBP41cR8y3NyDuIHXVnJlY1rDV8PUNbmtANLUbakW9NRTnmMHLzxa3KXAOJoUwYFxbbtUflUmvP; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WFUHoUkz0PekjTvoM.HlSOx5JpX5KzhUgL.Foqp1h-7e0epe0M2dJLoIEXLxKBLBo.L12eLxK.LB.-L1K.LxKnL12eLBoqLxKML1K-LB-2LxK.L1K-LB.qt; ALF=02_1769354157; WBPSESS=2JbmQMfDBf9GhITJyWUUWznL60fHFOFR2V0qqV--Q6QQ6CjSe-HiZ0xa9TFn-LDS5O1o4Bt-6sV17vRiK-t7ZlO0yRTJy9-qLgMfxKRseCi4GSNZdG28OabcKbLXwnlOjrXyAqjKEgvO3nDv1a5IwQ=="
    uid = "3197845214"
    
    # 测试cookies
    if not test_cookies(cookies):
        get_fresh_cookies_guide()
        return
    
    # 测试用户API
    success, working_api = test_user_api(cookies, uid)
    
    if success:
        print("\n" + "="*60)
        print("✅ 测试通过！可以开始爬取")
        print("="*60)
        print(f"\n可用的API: {working_api}")
        print(f"\n下一步: 修改 weibo_crawler.py 使用这个API端点")
        
    else:
        print("\n" + "="*60)
        print("❌ 测试失败")
        print("="*60)
        print("\n可能的原因:")
        print("1. Cookie已过期 - 需要重新获取")
        print("2. 用户ID不存在或隐私设置限制")
        print("3. 被微博的反爬虫机制拦截")
        print("4. 网络连接问题")
        print("\n建议:")
        print("1. 重新获取最新的Cookie（参考下面的指南）")
        print("2. 尝试使用你自己的微博UID")
        print("3. 考虑使用GitHub数据源（更简单可靠）")
        
        get_fresh_cookies_guide()
        
        print("\n" + "="*60)
        print("💡 推荐替代方案")
        print("="*60)
        print("""
微博爬虫比较复杂且容易失败，我强烈建议使用以下替代方案:

方案1: GitHub爬虫 (最推荐)
  cd /home/honglianglu/hdd/deanony
  python step1_generate_data.py  # 或使用GitHub API

方案2: 公开数据集
  下载SNAP的Facebook/Twitter数据集
  
方案3: 示例数据
  ./run_all.sh  # 使用生成的示例数据

这些方案都能完成你的大作业，效果不会比微博差！
        """)


if __name__ == "__main__":
    main()


