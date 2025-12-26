"""
使用weibo-crawler库的现代化微博爬虫
这是基于2024年最新方法的改进版本
"""

import json
import time
import sys
from pathlib import Path
from tqdm import tqdm
import logging

# 解决本地文件名冲突：优先从site-packages导入
import site
for site_dir in site.getsitepackages():
    if site_dir not in sys.path:
        sys.path.insert(0, site_dir)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 首先需要安装: pip install weibo-crawler
# 注意：包名是weibo-crawler，但导入时用weibo_crawler
try:
    from weibo_crawler import Profile, Follow
    WEIBO_CRAWLER_AVAILABLE = True
    logger.info("✅ weibo-crawler库已加载")
except ImportError as e:
    WEIBO_CRAWLER_AVAILABLE = False
    logger.warning(f"weibo-crawler未安装: {e}")


class ModernWeiboCrawler:
    """使用weibo-crawler库的现代化爬虫"""
    
    def __init__(self, cookies: str):
        """
        初始化爬虫
        
        Args:
            cookies: 微博cookies字符串
        """
        if not WEIBO_CRAWLER_AVAILABLE:
            raise ImportError("请先安装: pip install weibo-crawler")
        
        self.cookies = cookies
        
        # weibo-crawler需要指定CSV文件路径
        data_dir = Path(__file__).parent.parent / "data" / "raw"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        profile_csv = str(data_dir / "weibo_profiles.csv")
        follow_csv = str(data_dir / "weibo_follows.csv")
        
        self.profile = Profile(cookies=cookies, csvfile=profile_csv)
        self.follow = Follow(cookies=cookies, csvfile=follow_csv)
    
    def get_user_info(self, uid: str) -> dict:
        """
        获取用户信息
        
        Args:
            uid: 用户ID
            
        Returns:
            用户信息字典
        """
        try:
            user_data = self.profile.get_profile(userid=uid)
            
            if user_data:
                return {
                    'uid': uid,
                    'screen_name': user_data.get('screen_name', ''),
                    'followers_count': user_data.get('followers_count', 0),
                    'follow_count': user_data.get('follow_count', 0),
                    'description': user_data.get('description', ''),
                }
            else:
                logger.warning(f"无法获取用户 {uid} 的信息")
                return None
                
        except Exception as e:
            logger.error(f"获取用户 {uid} 信息时出错: {e}")
            return None
    
    def get_followings(self, uid: str, max_count: int = 100) -> list:
        """
        获取用户关注列表
        
        Args:
            uid: 用户ID
            max_count: 最大获取数量
            
        Returns:
            关注的用户ID列表
        """
        try:
            followings = self.follow.get_followings(userid=uid)
            
            if followings and isinstance(followings, list):
                # 提取用户ID
                following_uids = []
                for user in followings[:max_count]:
                    if isinstance(user, dict) and 'id' in user:
                        following_uids.append(str(user['id']))
                    elif isinstance(user, str):
                        following_uids.append(user)
                
                return following_uids
            else:
                return []
                
        except Exception as e:
            logger.error(f"获取用户 {uid} 关注列表时出错: {e}")
            return []
    
    def get_followers(self, uid: str, max_count: int = 100) -> list:
        """
        获取用户粉丝列表
        
        Args:
            uid: 用户ID
            max_count: 最大获取数量
            
        Returns:
            粉丝用户ID列表
        """
        try:
            followers = self.follow.get_followers(userid=uid)
            
            if followers and isinstance(followers, list):
                follower_uids = []
                for user in followers[:max_count]:
                    if isinstance(user, dict) and 'id' in user:
                        follower_uids.append(str(user['id']))
                    elif isinstance(user, str):
                        follower_uids.append(user)
                
                return follower_uids
            else:
                return []
                
        except Exception as e:
            logger.error(f"获取用户 {uid} 粉丝列表时出错: {e}")
            return []
    
    def crawl_network(self, start_uid: str, max_users: int = 1000,
                     max_depth: int = 2, delay: float = 2.0) -> dict:
        """
        BFS爬取社交网络
        
        Args:
            start_uid: 起始用户ID
            max_users: 最大用户数
            max_depth: 最大深度
            delay: 请求间隔（秒），建议>=2秒避免被封
            
        Returns:
            包含用户信息和关注关系的字典
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
                continue
            
            users[uid] = user_info
            pbar.update(1)
            
            # 添加延迟避免被封
            time.sleep(delay)
            
            # 获取关注列表
            if depth < max_depth:
                followings = self.get_followings(uid, max_count=50)
                
                for following_uid in followings:
                    edges.append((uid, following_uid))
                    
                    if following_uid not in visited and len(users) < max_users:
                        queue.append((following_uid, depth + 1))
                
                # 再次延迟
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
        
        logger.info(f"数据已保存到: {output_path}")


def test_weibo_crawler():
    """测试weibo-crawler库是否可用"""
    print("="*60)
    print("测试weibo-crawler库")
    print("="*60)
    
    if not WEIBO_CRAWLER_AVAILABLE:
        print("\n❌ weibo-crawler库未安装")
        print("\n安装方法:")
        print("  pip install weibo-crawler")
        print("\n或者:")
        print("  pip install weibo-crawler -i https://pypi.tuna.tsinghua.edu.cn/simple")
        return False
    
    print("\n✅ weibo-crawler库已安装")
    print("\n可用的类:")
    print("  - Profile: 获取用户信息")
    print("  - Follow: 获取关注/粉丝")
    print("  - Weibos: 获取微博内容")
    
    return True


def main():
    """主函数"""
    # 测试库是否可用
    if not test_weibo_crawler():
        print("\n" + "="*60)
        print("💡 替代方案")
        print("="*60)
        print("""
由于微博反爬虫机制较强，建议使用以下替代方案:

方案1: 使用示例数据（最快）
  cd /home/honglianglu/hdd/deanony
  ./run_all.sh

方案2: 使用GitHub数据（推荐）
  python step1_generate_data.py  # 改用GitHub

方案3: 使用公开数据集
  下载SNAP的Facebook/Twitter数据集
        """)
        return
    
    # 你的cookies（需要替换为最新的）
    cookies = "SINAGLOBAL=6740185828856.008.1764257392979; XSRF-TOKEN=Myn4TmTnG35cSjgyYPIJfvmV; SCF=AjiMSHwPp3pk5eVrMx10d6WYKiUi8q5VEC2hifoXmNfxm-mQDE2IPwP4DaI7i_6W3iyQ4sat5D1N02_MdRCywNM.; SUB=_2A25EStb9DeRhGeBP41cR8y3NyDuIHXVnJlY1rDV8PUNbmtANLUbakW9NRTnmMHLzxa3KXAOJoUwYFxbbtUflUmvP; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WFUHoUkz0PekjTvoM.HlSOx5JpX5KzhUgL.Foqp1h-7e0epe0M2dJLoIEXLxKBLBo.L12eLxK.LB.-L1K.LxKnL12eLBoqLxKML1K-LB-2LxK.L1K-LB.qt; ALF=02_1769354157; _s_tentry=weibo.com; Apache=7516984194176.165.1766762537792; ULV=1766762537794:2:1:1:7516984194176.165.1766762537792:1764257392981; WBPSESS=2JbmQMfDBf9GhITJyWUUWznL60fHFOFR2V0qqV--Q6QQ6CjSe-HiZ0xa9TFn-LDSQqDlY1BaeyAkFOeWqX_zXuy2IQtbUl_bkq6V5XSWjW4mXeVHy0BlQrpgbFODloUw3x_fxkG6hoMnOUUDzYCjtA=="
    
    start_uid = "6185033137"
    
    print("\n开始爬取微博数据...")
    print(f"起始用户: {start_uid}")
    print(f"注意: 请求间隔2秒，避免被封IP")
    
    try:
        crawler = ModernWeiboCrawler(cookies=cookies)
        
        # 先测试单个用户
        print("\n测试获取用户信息...")
        user_info = crawler.get_user_info(start_uid)
        
        if user_info:
            print(f"✅ 用户信息获取成功:")
            print(f"   昵称: {user_info.get('screen_name', 'N/A')}")
            print(f"   粉丝数: {user_info.get('followers_count', 0)}")
            print(f"   关注数: {user_info.get('follow_count', 0)}")
            
            # 开始完整爬取
            print("\n开始爬取社交网络...")
            data = crawler.crawl_network(
                start_uid=start_uid,
                max_users=500,  # 先爬500个测试
                max_depth=2,
                delay=2.0  # 2秒延迟
            )
            
            # 保存数据
            output_path = Path(__file__).parent.parent / "data" / "raw" / "weibo_data.json"
            crawler.save_data(data, output_path)
            
            print(f"\n✅ 爬取完成!")
            print(f"📊 用户数: {data['metadata']['total_users']}")
            print(f"📊 关系数: {data['metadata']['total_edges']}")
            
        else:
            print("\n❌ 无法获取用户信息")
            print("\n可能的原因:")
            print("1. Cookie已过期，需要重新获取")
            print("2. 用户ID不存在")
            print("3. 该用户设置了隐私保护")
            print("\n获取最新Cookie的方法:")
            print("1. 访问 https://m.weibo.cn")
            print("2. 登录你的账号")
            print("3. F12 → Network → 刷新页面")
            print("4. 找到任意请求 → Request Headers → Cookie")
            print("5. 完整复制Cookie值")
            
    except Exception as e:
        logger.error(f"爬取过程中出错: {e}")
        print("\n" + "="*60)
        print("💡 建议")
        print("="*60)
        print("微博爬虫比较复杂，建议使用替代方案：")
        print("  ./run_all.sh  # 使用示例数据")


if __name__ == "__main__":
    main()

