"""
改进版微博爬虫 - 使用多种方法获取关注列表
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import json
from pathlib import Path
from tqdm import tqdm
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImprovedWeiboCrawler:
    """改进版微博爬虫"""
    
    def __init__(self, headless=False):
        logger.info("正在启动Chrome浏览器...")
        
        chrome_options = Options()
        
        if headless:
            chrome_options.add_argument('--headless')
        
        # 反爬虫设置
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # 模拟真实浏览器
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36')
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.wait = WebDriverWait(self.driver, 10)
        
        # 移除webdriver标识
        self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
            'source': '''
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                })
            '''
        })
        
        logger.info("✅ 浏览器启动成功!")
    
    def manual_login(self):
        """手动登录"""
        logger.info("请在浏览器中手动登录微博...")
        
        self.driver.get('https://weibo.com/')
        time.sleep(2)
        
        print("\n" + "="*70)
        print("请在打开的浏览器窗口中:")
        print("  1. 点击登录按钮")
        print("  2. 输入账号密码")
        print("  3. 完成登录（包括滑块验证等）")
        print("  4. 看到微博首页后")
        print("="*70)
        
        input("\n登录完成后，按回车键继续...")
        
        logger.info("✅ 继续执行...")
        return True
    
    def get_user_info(self, uid: str) -> dict:
        """获取用户信息"""
        url = f'https://weibo.com/u/{uid}'
        
        try:
            self.driver.get(url)
            time.sleep(3)
            
            # 获取昵称
            try:
                nickname = self.driver.find_element(By.CSS_SELECTOR, '[class*="head_nick"]').text
            except:
                nickname = self.driver.title.split('-')[0].strip() if '-' in self.driver.title else f'user_{uid}'
            
            logger.info(f"✅ 获取用户: {nickname}")
            
            return {
                'uid': uid,
                'screen_name': nickname,
                'followers_count': 0,
                'follow_count': 0,
                'description': ''
            }
            
        except Exception as e:
            logger.error(f"❌ 获取用户 {uid} 失败: {e}")
            return None
    
    def get_followings_method1(self, uid: str, max_count: int = 50) -> list:
        """方法1: 从关注页面直接解析"""
        followings = []
        
        url = f'https://weibo.com/u/{uid}/follow'
        logger.info(f"  方法1: 访问关注页面 {url}")
        
        try:
            self.driver.get(url)
            time.sleep(3)
            
            # 尝试多次滚动加载
            for i in range(5):
                # 滚动到底部
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                
                # 查找所有用户链接
                try:
                    # 尝试不同的选择器
                    selectors = [
                        'a[href*="/u/"]',
                        'a[href*="/profile/"]',
                        '[class*="card"] a[href*="/u/"]'
                    ]
                    
                    for selector in selectors:
                        links = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        
                        for link in links:
                            try:
                                href = link.get_attribute('href')
                                if href and '/u/' in href:
                                    # 提取UID
                                    match = re.search(r'/u/(\d+)', href)
                                    if match:
                                        following_uid = match.group(1)
                                        if following_uid != uid and following_uid not in followings:
                                            followings.append(following_uid)
                                            
                                            if len(followings) >= max_count:
                                                logger.info(f"  ✅ 方法1获取到 {len(followings)} 个用户")
                                                return followings
                            except:
                                continue
                    
                    if len(followings) > 0:
                        logger.info(f"  已获取 {len(followings)} 个用户...")
                
                except Exception as e:
                    logger.debug(f"  滚动 {i+1} 出错: {e}")
                    continue
            
            logger.info(f"  方法1获取到 {len(followings)} 个用户")
            return followings
            
        except Exception as e:
            logger.error(f"  方法1失败: {e}")
            return []
    
    def get_followings_method2(self, uid: str, max_count: int = 50) -> list:
        """方法2: 使用移动端页面"""
        followings = []
        
        url = f'https://m.weibo.cn/u/{uid}'
        logger.info(f"  方法2: 访问移动端 {url}")
        
        try:
            # 临时切换到移动端UA
            self.driver.execute_cdp_cmd('Network.setUserAgentOverride', {
                "userAgent": 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)'
            })
            
            self.driver.get(url)
            time.sleep(3)
            
            # 查找关注按钮并点击
            try:
                follow_buttons = self.driver.find_elements(By.XPATH, "//*[contains(text(), '关注')]")
                if follow_buttons:
                    follow_buttons[0].click()
                    time.sleep(2)
            except:
                pass
            
            # 解析用户列表
            for i in range(3):
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                
                # 查找用户卡片
                try:
                    cards = self.driver.find_elements(By.CSS_SELECTOR, '[class*="card"]')
                    for card in cards:
                        try:
                            links = card.find_elements(By.TAG_NAME, 'a')
                            for link in links:
                                href = link.get_attribute('href')
                                if href and '/u/' in href:
                                    match = re.search(r'/u/(\d+)', href)
                                    if match:
                                        following_uid = match.group(1)
                                        if following_uid != uid and following_uid not in followings:
                                            followings.append(following_uid)
                        except:
                            continue
                except:
                    pass
            
            # 恢复PC UA
            self.driver.execute_cdp_cmd('Network.setUserAgentOverride', {
                "userAgent": 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            })
            
            logger.info(f"  方法2获取到 {len(followings)} 个用户")
            return followings[:max_count]
            
        except Exception as e:
            logger.error(f"  方法2失败: {e}")
            return []
    
    def get_followings_method3(self, uid: str, max_count: int = 50) -> list:
        """方法3: 从用户主页提取互动用户"""
        followings = []
        
        url = f'https://weibo.com/u/{uid}'
        logger.info(f"  方法3: 从主页提取互动用户")
        
        try:
            self.driver.get(url)
            time.sleep(3)
            
            # 滚动加载微博
            for i in range(10):
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1.5)
                
                # 查找所有@用户 和 转发用户
                try:
                    # 查找所有包含用户链接的元素
                    all_links = self.driver.find_elements(By.CSS_SELECTOR, 'a[href*="/u/"], a[href*="/n/"]')
                    
                    for link in all_links:
                        try:
                            href = link.get_attribute('href')
                            if href and '/u/' in href:
                                match = re.search(r'/u/(\d+)', href)
                                if match:
                                    other_uid = match.group(1)
                                    if other_uid != uid and other_uid not in followings:
                                        followings.append(other_uid)
                                        
                                        if len(followings) >= max_count:
                                            logger.info(f"  ✅ 方法3获取到 {len(followings)} 个用户")
                                            return followings
                        except:
                            continue
                
                except Exception as e:
                    continue
            
            logger.info(f"  方法3获取到 {len(followings)} 个用户")
            return followings
            
        except Exception as e:
            logger.error(f"  方法3失败: {e}")
            return []
    
    def get_followings(self, uid: str, max_count: int = 50) -> list:
        """获取关注列表 - 尝试所有方法"""
        logger.info(f"正在获取用户 {uid} 的关注列表...")
        
        # 依次尝试3种方法
        all_followings = []
        
        # 方法1: 从关注页面
        followings1 = self.get_followings_method1(uid, max_count)
        all_followings.extend(followings1)
        
        if len(all_followings) >= max_count:
            logger.info(f"✅ 共获取 {len(all_followings[:max_count])} 个关注用户")
            return all_followings[:max_count]
        
        # 方法2: 移动端
        followings2 = self.get_followings_method2(uid, max_count - len(all_followings))
        all_followings.extend([f for f in followings2 if f not in all_followings])
        
        if len(all_followings) >= max_count:
            logger.info(f"✅ 共获取 {len(all_followings[:max_count])} 个关注用户")
            return all_followings[:max_count]
        
        # 方法3: 从主页互动用户
        followings3 = self.get_followings_method3(uid, max_count - len(all_followings))
        all_followings.extend([f for f in followings3 if f not in all_followings])
        
        # 去重
        all_followings = list(set(all_followings))
        
        logger.info(f"✅ 共获取 {len(all_followings)} 个关注用户（组合3种方法）")
        return all_followings[:max_count]
    
    def crawl_network(self, start_uid: str, max_users: int = 200, 
                     max_depth: int = 2, delay: float = 3.0) -> dict:
        """BFS爬取社交网络"""
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
            
            # 获取关注列表
            if depth < max_depth:
                followings = self.get_followings(uid, max_count=20)
                
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
    
    def close(self):
        """关闭浏览器"""
        if self.driver:
            self.driver.quit()
            logger.info("浏览器已关闭")


def main():
    print("="*70)
    print("改进版微博爬虫 - 使用多种方法获取关注列表")
    print("="*70)
    
    # 配置
    start_uid = input("\n请输入起始用户UID（直接回车使用'人民日报'）: ").strip()
    if not start_uid:
        start_uid = "2803301701"
        tqdm.write(f"使用默认: 人民日报 ({start_uid})")
    
    max_users_input = input("\n爬取多少个用户？（直接回车使用200）: ").strip()
    max_users = int(max_users_input) if max_users_input else 200
    
    show_browser = input("\n是否显示浏览器窗口？(y/n，默认y): ").strip().lower()
    headless = (show_browser == 'n')
    
    # 创建爬虫
    try:
        crawler = ImprovedWeiboCrawler(headless=headless)
        
        # 手动登录
        crawler.manual_login()
        
        # 开始爬取
        print("\n" + "="*70)
        print("开始爬取社交网络")
        print("="*70)
        print(f"起始用户: {start_uid}")
        print(f"目标数量: {max_users} 个用户")
        print("\n⚠️  请不要关闭浏览器窗口!\n")
        
        data = crawler.crawl_network(
            start_uid=start_uid,
            max_users=max_users,
            max_depth=2,
            delay=3.0
        )
        
        # 保存数据
        output_path = Path('data/raw/weibo_improved_data.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print("\n" + "="*70)
        print("🎉 爬取完成!")
        print("="*70)
        print(f"用户数: {data['metadata']['total_users']}")
        print(f"关系数: {data['metadata']['total_edges']}")
        print(f"数据文件: {output_path}")
        
        crawler.close()
        
        if data['metadata']['total_edges'] > 0:
            tqdm.write("\n✅ 成功获取到关注关系！")
            tqdm.write("\n下一步:")
            tqdm.write(f"  1. python step2_build_graph.py --input {output_path}")
            tqdm.write("  2. python step3_anonymize.py")
            tqdm.write("  3. python step4_attack.py")
        else:
            tqdm.write("\n⚠️  未获取到关注关系")
            tqdm.write("可能原因：")
            tqdm.write("  1. 未正确登录")
            tqdm.write("  2. 该用户没有公开的关注列表")
            tqdm.write("  3. 页面结构变化")
            tqdm.write("\n建议: 尝试使用你自己的微博账号UID")
        
    except Exception as e:
        logger.error(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        try:
            crawler.close()
        except:
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")

