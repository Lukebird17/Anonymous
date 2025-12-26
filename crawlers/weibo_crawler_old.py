"""
微博爬虫模块
用于爬取用户关注关系构建社交网络图

使用说明:
1. 需要登录微博获取cookies
2. 建议使用移动端API (m.weibo.cn) 反爬较弱
3. 注意请求频率，避免被封IP
"""

import requests
import json
import time
import random
from typing import Dict, List, Set
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeiboCrawler:
    """微博爬虫类"""
    
    def __init__(self, cookies: str = None, delay: float = 1.0):
        """
        初始化微博爬虫
        
        Args:
            cookies: 登录后的cookies字符串
            delay: 请求间隔时间（秒）
        """
        self.session = requests.Session()
        self.delay = delay
        self.base_url = "https://m.weibo.cn/api"
        
        # 设置headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15',
            'Referer': 'https://m.weibo.cn/',
        })
        
        if cookies:
            self.session.headers.update({'Cookie': cookies})
    
    def get_user_info(self, uid: str) -> Dict:
        """
        获取用户基本信息
        
        Args:
            uid: 用户ID
            
        Returns:
            用户信息字典
        """
        url = f"{self.base_url}/container/getIndex"
        params = {
            'type': 'uid',
            'value': uid,
            'containerid': f'100505{uid}'
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('ok') == 1:
                user_info = data.get('data', {}).get('userInfo', {})
                return {
                    'uid': uid,
                    'screen_name': user_info.get('screen_name', ''),
                    'followers_count': user_info.get('followers_count', 0),
                    'follow_count': user_info.get('follow_count', 0),
                    'description': user_info.get('description', ''),
                }
            else:
                logger.warning(f"获取用户 {uid} 信息失败: {data.get('msg', 'Unknown error')}")
                return None
                
        except Exception as e:
            logger.error(f"获取用户 {uid} 信息时出错: {e}")
            return None
    
    def get_followings(self, uid: str, max_page: int = 5) -> List[str]:
        """
        获取用户的关注列表
        
        Args:
            uid: 用户ID
            max_page: 最大页数
            
        Returns:
            关注的用户ID列表
        """
        followings = []
        url = f"{self.base_url}/container/getIndex"
        
        for page in range(1, max_page + 1):
            params = {
                'containerid': f'231051_-_followers_-_{uid}',
                'page': page
            }
            
            try:
                time.sleep(self.delay + random.uniform(0, 0.5))
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if data.get('ok') != 1:
                    break
                
                cards = data.get('data', {}).get('cards', [])
                if not cards:
                    break
                
                for card in cards:
                    card_group = card.get('card_group', [])
                    for user in card_group:
                        user_data = user.get('user', {})
                        if 'id' in user_data:
                            followings.append(str(user_data['id']))
                
            except Exception as e:
                logger.error(f"获取用户 {uid} 关注列表时出错 (page {page}): {e}")
                break
        
        return followings
    
    def get_followers(self, uid: str, max_page: int = 5) -> List[str]:
        """
        获取用户的粉丝列表
        
        Args:
            uid: 用户ID
            max_page: 最大页数
            
        Returns:
            粉丝的用户ID列表
        """
        # 实现类似get_followings的逻辑
        # 这里为了简化，返回空列表，实际项目中应实现完整逻辑
        return []
    
    def crawl_network(self, start_uid: str, max_users: int = 5000, 
                     max_depth: int = 3) -> Dict:
        """
        BFS爬取社交网络
        
        Args:
            start_uid: 起始用户ID
            max_users: 最大用户数
            max_depth: 最大深度
            
        Returns:
            包含用户信息和关注关系的字典
        """
        users = {}
        edges = []
        visited = set()
        queue = [(start_uid, 0)]  # (uid, depth)
        
        pbar = tqdm(total=max_users, desc="爬取微博用户")
        
        while queue and len(users) < max_users:
            uid, depth = queue.pop(0)
            
            if uid in visited or depth > max_depth:
                continue
            
            visited.add(uid)
            
            # 获取用户信息
            user_info = self.get_user_info(uid)
            if not user_info:
                continue
            
            users[uid] = user_info
            pbar.update(1)
            
            # 获取关注列表
            if depth < max_depth:
                followings = self.get_followings(uid, max_page=3)
                
                for following_uid in followings:
                    edges.append((uid, following_uid))
                    
                    if following_uid not in visited and len(users) < max_users:
                        queue.append((following_uid, depth + 1))
        
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
    
    def save_data(self, data: Dict, output_path: Path):
        """
        保存爬取的数据
        
        Args:
            data: 爬取的数据
            output_path: 输出文件路径
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据已保存到: {output_path}")


def main():
    """主函数示例"""
    # 注意：需要替换为实际的cookies
    cookies = "SINAGLOBAL=6740185828856.008.1764257392979; ULV=1764257392981:1:1:1:6740185828856.008.1764257392979:; XSRF-TOKEN=Myn4TmTnG35cSjgyYPIJfvmV; SCF=AjiMSHwPp3pk5eVrMx10d6WYKiUi8q5VEC2hifoXmNfxm-mQDE2IPwP4DaI7i_6W3iyQ4sat5D1N02_MdRCywNM.; SUB=_2A25EStb9DeRhGeBP41cR8y3NyDuIHXVnJlY1rDV8PUNbmtANLUbakW9NRTnmMHLzxa3KXAOJoUwYFxbbtUflUmvP; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WFUHoUkz0PekjTvoM.HlSOx5JpX5KzhUgL.Foqp1h-7e0epe0M2dJLoIEXLxKBLBo.L12eLxK.LB.-L1K.LxKnL12eLBoqLxKML1K-LB-2LxK.L1K-LB.qt; ALF=02_1769354157; WBPSESS=2JbmQMfDBf9GhITJyWUUWznL60fHFOFR2V0qqV--Q6QQ6CjSe-HiZ0xa9TFn-LDS5O1o4Bt-6sV17vRiK-t7ZlO0yRTJy9-qLgMfxKRseCi4GSNZdG28OabcKbLXwnlOjrXyAqjKEgvO3nDv1a5IwQ=="
    
    crawler = WeiboCrawler(cookies=cookies, delay=1.0)
    
    # 从一个种子用户开始爬取
    # 注意：需要替换为实际的用户ID
    start_uid = "3197845214"
    
    data = crawler.crawl_network(
        start_uid=start_uid,
        max_users=5000,
        max_depth=3
    )
    
    # 保存数据
    output_path = Path(__file__).parent.parent / "data" / "raw" / "weibo_data.json"
    crawler.save_data(data, output_path)
    
    print(f"\n爬取完成!")
    print(f"用户数: {data['metadata']['total_users']}")
    print(f"关系数: {data['metadata']['total_edges']}")


if __name__ == "__main__":
    main()


