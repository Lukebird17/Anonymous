"""
GitHub爬虫模块
用于爬取GitHub开发者的Follow关系和Star/Fork关系

使用说明:
1. 需要GitHub Personal Access Token
2. API限流: 5000次/小时(认证用户)
3. 推荐爬取特定技术社区(如Rust/Go)
"""

import requests
import json
import time
from typing import Dict, List, Set
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GitHubCrawler:
    """GitHub爬虫类"""
    
    def __init__(self, token: str = None, delay: float = 0.5):
        """
        初始化GitHub爬虫
        
        Args:
            token: GitHub Personal Access Token
            delay: 请求间隔时间（秒）
        """
        self.session = requests.Session()
        self.delay = delay
        self.base_url = "https://api.github.com"
        
        # 设置headers
        headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'GitHub-Network-Crawler'
        }
        
        if token:
            headers['Authorization'] = f'token {token}'
        
        self.session.headers.update(headers)
    
    def check_rate_limit(self):
        """检查API限流状态"""
        url = f"{self.base_url}/rate_limit"
        try:
            response = self.session.get(url)
            data = response.json()
            remaining = data['rate']['remaining']
            reset_time = data['rate']['reset']
            
            if remaining < 100:
                wait_time = reset_time - time.time()
                if wait_time > 0:
                    logger.warning(f"API限流剩余次数较少 ({remaining})，等待 {wait_time:.0f} 秒")
                    time.sleep(wait_time)
        except Exception as e:
            logger.error(f"检查API限流时出错: {e}")
    
    def get_user_info(self, username: str) -> Dict:
        """
        获取用户基本信息
        
        Args:
            username: GitHub用户名
            
        Returns:
            用户信息字典
        """
        url = f"{self.base_url}/users/{username}"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                'username': username,
                'id': data.get('id'),
                'name': data.get('name', ''),
                'followers': data.get('followers', 0),
                'following': data.get('following', 0),
                'public_repos': data.get('public_repos', 0),
                'bio': data.get('bio', ''),
                'company': data.get('company', ''),
                'location': data.get('location', ''),
            }
            
        except Exception as e:
            logger.error(f"获取用户 {username} 信息时出错: {e}")
            return None
    
    def get_following(self, username: str, max_count: int = 100) -> List[str]:
        """
        获取用户的关注列表
        
        Args:
            username: GitHub用户名
            max_count: 最大获取数量
            
        Returns:
            关注的用户名列表
        """
        following = []
        page = 1
        per_page = 100
        
        while len(following) < max_count:
            url = f"{self.base_url}/users/{username}/following"
            params = {'page': page, 'per_page': per_page}
            
            try:
                time.sleep(self.delay)
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                
                following.extend([user['login'] for user in data])
                page += 1
                
                if len(data) < per_page:
                    break
                    
            except Exception as e:
                logger.error(f"获取用户 {username} 关注列表时出错: {e}")
                break
        
        return following[:max_count]
    
    def get_followers(self, username: str, max_count: int = 100) -> List[str]:
        """
        获取用户的粉丝列表
        
        Args:
            username: GitHub用户名
            max_count: 最大获取数量
            
        Returns:
            粉丝用户名列表
        """
        followers = []
        page = 1
        per_page = 100
        
        while len(followers) < max_count:
            url = f"{self.base_url}/users/{username}/followers"
            params = {'page': page, 'per_page': per_page}
            
            try:
                time.sleep(self.delay)
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                
                followers.extend([user['login'] for user in data])
                page += 1
                
                if len(data) < per_page:
                    break
                    
            except Exception as e:
                logger.error(f"获取用户 {username} 粉丝列表时出错: {e}")
                break
        
        return followers[:max_count]
    
    def get_starred_repos(self, username: str, max_count: int = 50) -> List[str]:
        """
        获取用户Star的仓库列表
        
        Args:
            username: GitHub用户名
            max_count: 最大获取数量
            
        Returns:
            仓库全名列表 (owner/repo)
        """
        starred = []
        page = 1
        per_page = 100
        
        while len(starred) < max_count:
            url = f"{self.base_url}/users/{username}/starred"
            params = {'page': page, 'per_page': per_page}
            
            try:
                time.sleep(self.delay)
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                
                starred.extend([repo['full_name'] for repo in data])
                page += 1
                
                if len(data) < per_page:
                    break
                    
            except Exception as e:
                logger.error(f"获取用户 {username} Star列表时出错: {e}")
                break
        
        return starred[:max_count]
    
    def search_users_by_language(self, language: str, max_users: int = 100) -> List[str]:
        """
        按编程语言搜索用户（用于找种子用户）
        
        Args:
            language: 编程语言（如rust, go, python）
            max_users: 最大用户数
            
        Returns:
            用户名列表
        """
        users = []
        page = 1
        per_page = 100
        
        while len(users) < max_users:
            url = f"{self.base_url}/search/users"
            params = {
                'q': f'language:{language}',
                'sort': 'followers',
                'order': 'desc',
                'page': page,
                'per_page': per_page
            }
            
            try:
                time.sleep(self.delay)
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                items = data.get('items', [])
                if not items:
                    break
                
                users.extend([user['login'] for user in items])
                page += 1
                
            except Exception as e:
                logger.error(f"搜索用户时出错: {e}")
                break
        
        return users[:max_users]
    
    def crawl_network(self, start_users: List[str] = None, 
                     language: str = None,
                     max_users: int = 5000, 
                     max_depth: int = 2) -> Dict:
        """
        BFS爬取GitHub社交网络
        
        Args:
            start_users: 起始用户列表
            language: 如果start_users为空，则搜索该语言的用户作为起点
            max_users: 最大用户数
            max_depth: 最大深度
            
        Returns:
            包含用户信息和关注关系的字典
        """
        # 获取种子用户
        if not start_users:
            if language:
                logger.info(f"搜索 {language} 社区的种子用户...")
                start_users = self.search_users_by_language(language, max_users=10)
            else:
                raise ValueError("必须提供start_users或language参数")
        
        users = {}
        edges = []
        starred_repos = {}  # 用于构建异构图
        visited = set()
        queue = [(username, 0) for username in start_users]
        
        pbar = tqdm(total=max_users, desc="爬取GitHub用户")
        
        while queue and len(users) < max_users:
            self.check_rate_limit()
            
            username, depth = queue.pop(0)
            
            if username in visited or depth > max_depth:
                continue
            
            visited.add(username)
            
            # 获取用户信息
            user_info = self.get_user_info(username)
            if not user_info:
                continue
            
            users[username] = user_info
            pbar.update(1)
            
            # 获取关注列表
            if depth < max_depth:
                following = self.get_following(username, max_count=100)
                
                for follow_username in following:
                    edges.append((username, follow_username, 'follow'))
                    
                    if follow_username not in visited and len(users) < max_users:
                        queue.append((follow_username, depth + 1))
            
            # 获取Star的仓库（用于构建异构图）
            repos = self.get_starred_repos(username, max_count=50)
            starred_repos[username] = repos
        
        pbar.close()
        
        return {
            'users': users,
            'edges': edges,
            'starred_repos': starred_repos,
            'metadata': {
                'start_users': start_users,
                'language': language,
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
    # 注意：需要替换为实际的GitHub Token
    token = "YOUR_GITHUB_TOKEN_HERE"
    
    crawler = GitHubCrawler(token=token, delay=0.5)
    
    # 爬取Rust社区的网络
    data = crawler.crawl_network(
        language="rust",
        max_users=5000,
        max_depth=2
    )
    
    # 保存数据
    output_path = Path(__file__).parent.parent / "data" / "raw" / "github_data.json"
    crawler.save_data(data, output_path)
    
    print(f"\n爬取完成!")
    print(f"用户数: {data['metadata']['total_users']}")
    print(f"关系数: {data['metadata']['total_edges']}")


if __name__ == "__main__":
    main()


