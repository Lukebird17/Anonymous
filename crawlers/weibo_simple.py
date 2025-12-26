"""
方法2: 直接使用requests + 正确的API端点
基于2024年最新的微博移动端API
"""

import requests
import json
import time
from pathlib import Path
from tqdm import tqdm


class SimpleWeiboCrawler:
    """简化版微博爬虫 - 直接请求API"""
    
    def __init__(self, cookies: str):
        """
        初始化
        
        Args:
            cookies: Cookie字符串
        """
        self.session = requests.Session()
        self.cookies_dict = self._parse_cookies(cookies)
        
        # 重要：设置完整的Headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148 Weibo (iPhone10,1__weibo__11.5.1__iphone__os14.0)',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Referer': 'https://m.weibo.cn/',
            'X-Requested-With': 'XMLHttpRequest',
        })
        
        self.session.cookies.update(self.cookies_dict)
    
    def _parse_cookies(self, cookie_str: str) -> dict:
        """解析Cookie字符串为字典"""
        cookies = {}
        for item in cookie_str.split('; '):
            if '=' in item:
                key, value = item.split('=', 1)
                cookies[key.strip()] = value.strip()
        return cookies
    
    def get_user_info(self, uid: str) -> dict:
        """
        获取用户信息 - 使用正确的API
        """
        # 方法1: 使用profile/info接口
        url1 = f'https://m.weibo.cn/profile/info?uid={uid}'
        
        # 方法2: 使用container接口
        url2 = f'https://m.weibo.cn/api/container/getIndex?type=uid&value={uid}'
        
        # 方法3: 使用用户主页
        url3 = f'https://m.weibo.cn/api/container/getIndex?containerid=100505{uid}'
        
        for url in [url1, url2, url3]:
            try:
                response = self.session.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # 提取用户信息
                    user_info = None
                    if 'data' in data and 'user' in data['data']:
                        user_info = data['data']['user']
                    elif 'data' in data and 'userInfo' in data['data']:
                        user_info = data['data']['userInfo']
                    
                    if user_info:
                        return {
                            'uid': uid,
                            'screen_name': user_info.get('screen_name', ''),
                            'followers_count': user_info.get('followers_count', 0),
                            'follow_count': user_info.get('follow_count', 0),
                            'description': user_info.get('description', ''),
                        }
            except Exception as e:
                print(f"尝试 {url} 失败: {e}")
                continue
        
        print(f"所有方法都失败了，无法获取用户 {uid}")
        return None
    
    def get_followings(self, uid: str, max_count: int = 50) -> list:
        """
        获取关注列表
        """
        followings = []
        page = 1
        
        while len(followings) < max_count:
            # 关注列表API
            url = f'https://m.weibo.cn/api/container/getIndex?containerid=231051_-_followers_-_{uid}&page={page}'
            
            try:
                time.sleep(2)  # 重要：延迟避免被封
                response = self.session.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get('ok') != 1:
                        break
                    
                    cards = data.get('data', {}).get('cards', [])
                    if not cards:
                        break
                    
                    for card in cards:
                        for item in card.get('card_group', []):
                            user = item.get('user', {})
                            if 'id' in user:
                                followings.append(str(user['id']))
                    
                    page += 1
                else:
                    print(f"请求失败，状态码: {response.status_code}")
                    break
                    
            except Exception as e:
                print(f"获取关注列表失败: {e}")
                break
        
        return followings[:max_count]
    
    def test_connection(self):
        """测试连接和Cookie是否有效"""
        print("测试微博连接...")
        
        # 测试1: 访问首页
        try:
            response = self.session.get('https://m.weibo.cn/', timeout=10)
            print(f"✅ 首页访问成功，状态码: {response.status_code}")
        except Exception as e:
            print(f"❌ 首页访问失败: {e}")
            return False
        
        # 测试2: 访问config接口
        try:
            response = self.session.get('https://m.weibo.cn/api/config', timeout=10)
            if response.status_code == 200:
                print(f"✅ API访问成功")
                return True
            else:
                print(f"❌ API访问失败，状态码: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ API测试失败: {e}")
            return False


def main():
    """测试新方法"""
    print("="*60)
    print("简化版微博爬虫测试")
    print("="*60)
    
    # 从你的文件中读取Cookie
    cookies = "SINAGLOBAL=6740185828856.008.1764257392979; XSRF-TOKEN=Myn4TmTnG35cSjgyYPIJfvmV; SCF=AjiMSHwPp3pk5eVrMx10d6WYKiUi8q5VEC2hifoXmNfxm-mQDE2IPwP4DaI7i_6W3iyQ4sat5D1N02_MdRCywNM.; SUB=_2A25EStb9DeRhGeBP41cR8y3NyDuIHXVnJlY1rDV8PUNbmtANLUbakW9NRTnmMHLzxa3KXAOJoUwYFxbbtUflUmvP; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WFUHoUkz0PekjTvoM.HlSOx5JpX5KzhUgL.Foqp1h-7e0epe0M2dJLoIEXLxKBLBo.L12eLxK.LB.-L1K.LxKnL12eLBoqLxKML1K-LB-2LxK.L1K-LB.qt; ALF=02_1769354157; _s_tentry=weibo.com; Apache=7516984194176.165.1766762537792; ULV=1766762537794:2:1:1:7516984194176.165.1766762537792:1764257392981; WBPSESS=2JbmQMfDBf9GhITJyWUUWznL60fHFOFR2V0qqV--Q6QQ6CjSe-HiZ0xa9TFn-LDSQqDlY1BaeyAkFOeWqX_zXuy2IQtbUl_bkq6V5XSWjW4mXeVHy0BlQrpgbFODloUw3x_fxkG6hoMnOUUDzYCjtA=="
    
    uid = "6185033137"
    
    crawler = SimpleWeiboCrawler(cookies)
    
    # 测试连接
    if crawler.test_connection():
        print("\n测试获取用户信息...")
        user_info = crawler.get_user_info(uid)
        
        if user_info and user_info.get('screen_name'):
            print(f"\n✅ 成功!")
            print(f"   昵称: {user_info['screen_name']}")
            print(f"   粉丝: {user_info['followers_count']}")
            print(f"   关注: {user_info['follow_count']}")
            
            print("\n测试获取关注列表...")
            followings = crawler.get_followings(uid, max_count=10)
            print(f"   获取到 {len(followings)} 个关注用户")
            print(f"   示例: {followings[:5]}")
            
        else:
            print("\n❌ 无法获取用户信息")
            print("可能原因:")
            print("1. Cookie已过期")
            print("2. 用户不存在或隐私设置")
            print("3. 被反爬虫拦截")
    else:
        print("\n连接测试失败，请检查Cookie")


if __name__ == "__main__":
    main()

