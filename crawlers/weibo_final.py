"""
微博爬虫 - 最终版本
使用多种方法确保能爬到真实数据
"""

import requests
import json
import time
from pathlib import Path
from tqdm import tqdm
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobustWeiboCrawler:
    """强力微博爬虫 - 多种备用方案"""
    
    def __init__(self, cookies: str):
        self.session = requests.Session()
        self.cookies_dict = self._parse_cookies(cookies)
        
        # 设置完整的请求头（模拟真实浏览器）
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://weibo.com/',
            'X-Requested-With': 'XMLHttpRequest',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
        })
        
        # 设置cookies
        for name, value in self.cookies_dict.items():
            self.session.cookies.set(name, value, domain='.weibo.com')
            self.session.cookies.set(name, value, domain='.weibo.cn')
    
    def _parse_cookies(self, cookie_str: str) -> dict:
        """解析Cookie"""
        cookies = {}
        for item in cookie_str.split('; '):
            if '=' in item:
                key, value = item.split('=', 1)
                cookies[key.strip()] = value.strip()
        return cookies
    
    def get_user_info_method1(self, uid: str) -> dict:
        """方法1: PC端API"""
        url = f'https://weibo.com/ajax/profile/info?uid={uid}'
        
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('ok') == 1 and 'data' in data:
                    user = data['data'].get('user', {})
                    if user.get('screen_name'):
                        logger.info(f"✅ 方法1成功获取用户: {user.get('screen_name')}")
                        return {
                            'uid': uid,
                            'screen_name': user.get('screen_name', ''),
                            'followers_count': user.get('followers_count', 0),
                            'follow_count': user.get('follow_count', 0),
                            'description': user.get('description', ''),
                        }
        except Exception as e:
            logger.debug(f"方法1失败: {e}")
        
        return None
    
    def get_user_info_method2(self, uid: str) -> dict:
        """方法2: 移动端API"""
        url = f'https://m.weibo.cn/api/container/getIndex?type=uid&value={uid}'
        
        try:
            # 临时修改UA为移动端
            old_ua = self.session.headers.get('User-Agent')
            self.session.headers['User-Agent'] = 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)'
            
            response = self.session.get(url, timeout=10)
            
            # 恢复UA
            self.session.headers['User-Agent'] = old_ua
            
            if response.status_code == 200:
                data = response.json()
                if data.get('ok') == 1:
                    user = data.get('data', {}).get('userInfo', {})
                    if user.get('screen_name'):
                        logger.info(f"✅ 方法2成功获取用户: {user.get('screen_name')}")
                        return {
                            'uid': uid,
                            'screen_name': user.get('screen_name', ''),
                            'followers_count': user.get('followers_count', 0),
                            'follow_count': user.get('follow_count', 0),
                            'description': user.get('description', ''),
                        }
        except Exception as e:
            logger.debug(f"方法2失败: {e}")
        
        return None
    
    def get_user_info_method3(self, uid: str) -> dict:
        """方法3: 直接访问用户主页HTML解析"""
        url = f'https://weibo.com/u/{uid}'
        
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                html = response.text
                
                # 尝试从HTML中提取用户信息
                # 查找$render_data
                match = re.search(r'\$render_data\s*=\s*(\[.*?\])\[0\]', html, re.DOTALL)
                if match:
                    try:
                        data_str = match.group(1)
                        data = json.loads(data_str)
                        if data and len(data) > 0:
                            user_data = data[0]
                            status = user_data.get('status', {})
                            user = status.get('user', {})
                            
                            if user.get('screen_name'):
                                logger.info(f"✅ 方法3成功获取用户: {user.get('screen_name')}")
                                return {
                                    'uid': uid,
                                    'screen_name': user.get('screen_name', ''),
                                    'followers_count': user.get('followers_count', 0),
                                    'follow_count': user.get('friends_count', 0),
                                    'description': user.get('description', ''),
                                }
                    except:
                        pass
        except Exception as e:
            logger.debug(f"方法3失败: {e}")
        
        return None
    
    def get_user_info(self, uid: str) -> dict:
        """获取用户信息 - 尝试所有方法"""
        logger.info(f"获取用户 {uid} 信息...")
        
        # 依次尝试3种方法
        for method in [self.get_user_info_method1, 
                      self.get_user_info_method2,
                      self.get_user_info_method3]:
            result = method(uid)
            if result:
                return result
            time.sleep(1)  # 方法之间延迟
        
        logger.warning(f"❌ 所有方法都无法获取用户 {uid}")
        return None
    
    def get_followings(self, uid: str, max_count: int = 50) -> list:
        """获取关注列表"""
        followings = []
        page = 1
        
        logger.info(f"获取用户 {uid} 的关注列表...")
        
        while len(followings) < max_count and page <= 5:
            # PC端关注API
            url = f'https://weibo.com/ajax/friendships/friends?uid={uid}&page={page}'
            
            try:
                time.sleep(2)  # 重要延迟
                response = self.session.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get('ok') == 1:
                        users = data.get('data', {}).get('users', [])
                        
                        if not users:
                            break
                        
                        for user in users:
                            if 'idstr' in user:
                                followings.append(user['idstr'])
                        
                        logger.info(f"  第{page}页: 获取到 {len(users)} 个用户")
                        page += 1
                    else:
                        logger.warning(f"  API返回错误: {data.get('msg', 'Unknown')}")
                        break
                else:
                    logger.warning(f"  HTTP错误: {response.status_code}")
                    break
                    
            except Exception as e:
                logger.error(f"  获取关注列表失败: {e}")
                break
        
        logger.info(f"  共获取 {len(followings)} 个关注用户")
        return followings[:max_count]
    
    def get_followers(self, uid: str, max_count: int = 50) -> list:
        """获取粉丝列表"""
        followers = []
        page = 1
        
        logger.info(f"获取用户 {uid} 的粉丝列表...")
        
        while len(followers) < max_count and page <= 5:
            url = f'https://weibo.com/ajax/friendships/followers?uid={uid}&page={page}'
            
            try:
                time.sleep(2)
                response = self.session.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get('ok') == 1:
                        users = data.get('data', {}).get('users', [])
                        
                        if not users:
                            break
                        
                        for user in users:
                            if 'idstr' in user:
                                followers.append(user['idstr'])
                        
                        logger.info(f"  第{page}页: 获取到 {len(users)} 个粉丝")
                        page += 1
                    else:
                        break
                else:
                    break
                    
            except Exception as e:
                logger.error(f"  获取粉丝列表失败: {e}")
                break
        
        logger.info(f"  共获取 {len(followers)} 个粉丝")
        return followers[:max_count]
    
    def crawl_network(self, start_uid: str, max_users: int = 500,
                     max_depth: int = 2, delay: float = 3.0) -> dict:
        """
        BFS爬取社交网络
        
        Args:
            start_uid: 起始用户ID
            max_users: 最大用户数
            max_depth: 最大深度
            delay: 请求间隔（秒）
        """
        users = {}
        edges = []
        visited = set()
        queue = [(start_uid, 0)]
        
        pbar = tqdm(total=max_users, desc="爬取微博用户")
        
        while queue and len(users) < max_users:
            uid, depth = queue.pop(0)
            
            if uid in visited or depth > max_depth:
                continue
            
            visited.add(uid)
            
            # 获取用户信息
            user_info = self.get_user_info(uid)
            if not user_info:
                logger.warning(f"跳过用户 {uid}")
                continue
            
            users[uid] = user_info
            pbar.update(1)
            
            time.sleep(delay)
            
            # 获取关注列表或粉丝列表
            if depth < max_depth:
                # 优先尝试获取关注列表
                followings = self.get_followings(uid, max_count=20)
                
                # 如果关注为0，则获取粉丝列表
                if len(followings) == 0:
                    logger.info(f"  用户{uid}关注数为0，改为获取粉丝列表")
                    followings = self.get_followers(uid, max_count=20)
                
                for following_uid in followings:
                    edges.append((uid, following_uid))
                    
                    if following_uid not in visited and len(users) < max_users:
                        queue.append((following_uid, depth + 1))
                
                time.sleep(delay)
        
        pbar.close()
        
        return {
            'users': users,
            'edges': edges,
            'metadata': {
                'start_uid': start_uid,
                'max_depth': max_depth,
                'total_users': len(users),
                'total_edges': len(edges)
            }
        }
    
    def save_data(self, data: dict, output_path: Path):
        """保存数据"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 数据已保存到: {output_path}")


def main():
    """主函数"""
    print("="*70)
    print("微博社交网络爬虫 - 最终版")
    print("="*70)
    
    # 你的cookies
    cookies = "SINAGLOBAL=6740185828856.008.1764257392979; XSRF-TOKEN=Myn4TmTnG35cSjgyYPIJfvmV; SCF=AjiMSHwPp3pk5eVrMx10d6WYKiUi8q5VEC2hifoXmNfxm-mQDE2IPwP4DaI7i_6W3iyQ4sat5D1N02_MdRCywNM.; SUB=_2A25EStb9DeRhGeBP41cR8y3NyDuIHXVnJlY1rDV8PUNbmtANLUbakW9NRTnmMHLzxa3KXAOJoUwYFxbbtUflUmvP; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WFUHoUkz0PekjTvoM.HlSOx5JpX5KzhUgL.Foqp1h-7e0epe0M2dJLoIEXLxKBLBo.L12eLxK.LB.-L1K.LxKnL12eLBoqLxKML1K-LB-2LxK.L1K-LB.qt; ALF=02_1769354157; _s_tentry=weibo.com; Apache=7516984194176.165.1766762537792; ULV=1766762537794:2:1:1:7516984194176.165.1766762537792:1764257392981; WBPSESS=2JbmQMfDBf9GhITJyWUUWznL60fHFOFR2V0qqV--Q6QQ6CjSe-HiZ0xa9TFn-LDSQqDlY1BaeyAkFOeWqX_zXuy2IQtbUl_bkq6V5XSWjW4mXeVHy0BlQrpgbFODloUw3x_fxkG6hoMnOUUDzYCjtA=="
    
    start_uid = "2803301701"  # 人民日报 - 有很多粉丝
    
    print(f"\n开始爬取...")
    print(f"起始用户: {start_uid}")
    print(f"目标数量: 500个用户")
    print(f"请求间隔: 3秒")
    print(f"\n⚠️  提示: 爬取过程较慢是正常的，避免被封IP\n")
    
    crawler = RobustWeiboCrawler(cookies)
    
    # 先测试单个用户
    print("="*70)
    print("测试: 获取起始用户信息")
    print("="*70)
    user_info = crawler.get_user_info(start_uid)
    
    if user_info and user_info.get('screen_name'):
        print(f"\n✅ 测试成功!")
        print(f"   昵称: {user_info['screen_name']}")
        print(f"   粉丝: {user_info['followers_count']}")
        print(f"   关注: {user_info['follow_count']}")
        
        # 继续爬取
        print("\n" + "="*70)
        print("开始爬取社交网络")
        print("="*70)
        
        data = crawler.crawl_network(
            start_uid=start_uid,
            max_users=500,
            max_depth=2,
            delay=3.0
        )
        
        # 保存数据
        output_path = Path(__file__).parent.parent / "data" / "raw" / "weibo_data.json"
        crawler.save_data(data, output_path)
        
        print(f"\n" + "="*70)
        print("爬取完成!")
        print("="*70)
        print(f"用户数: {data['metadata']['total_users']}")
        print(f"关系数: {data['metadata']['total_edges']}")
        print(f"数据文件: {output_path}")
        
    else:
        print("\n❌ 无法获取用户信息")
        print("\n可能的原因:")
        print("1. Cookie已过期（最可能）")
        print("2. 用户ID不存在")
        print("3. 该用户设置了隐私保护")
        print("4. 被微博的反爬虫系统拦截")
        
        print("\n💡 解决方法:")
        print("1. 重新登录微博获取最新Cookie")
        print("2. 尝试使用你自己的微博账号UID")
        print("3. 确保Cookie包含 SUB 和 SUBP 字段")


if __name__ == "__main__":
    main()

