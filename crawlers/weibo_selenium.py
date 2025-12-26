"""
方法1: 使用Selenium模拟浏览器爬取微博
这是最可靠的方法，可以绕过大部分反爬虫
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
import json
from pathlib import Path
from tqdm import tqdm


class SeleniumWeiboCrawler:
    """使用Selenium的微博爬虫"""
    
    def __init__(self, cookies_dict: dict = None):
        """
        初始化Selenium爬虫
        
        Args:
            cookies_dict: Cookie字典 {'name': 'value', ...}
        """
        # 配置Chrome选项
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # 无头模式
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.cookies_dict = cookies_dict
        self.wait = WebDriverWait(self.driver, 10)
    
    def login_with_cookies(self):
        """使用Cookies登录"""
        self.driver.get('https://m.weibo.cn')
        time.sleep(2)
        
        if self.cookies_dict:
            for name, value in self.cookies_dict.items():
                self.driver.add_cookie({'name': name, 'value': value})
            
            self.driver.refresh()
            time.sleep(2)
    
    def get_user_info(self, uid: str) -> dict:
        """获取用户信息"""
        url = f'https://m.weibo.cn/u/{uid}'
        self.driver.get(url)
        time.sleep(3)
        
        try:
            # 解析页面获取用户信息
            nickname_elem = self.driver.find_element(By.CLASS_NAME, 'lite-page-title')
            nickname = nickname_elem.text if nickname_elem else ''
            
            # 获取关注/粉丝数
            follow_info = self.driver.find_elements(By.CLASS_NAME, 'lite-iconf')
            followers = 0
            followings = 0
            
            if len(follow_info) >= 2:
                followings = follow_info[0].text.split()[0] if follow_info[0].text else '0'
                followers = follow_info[1].text.split()[0] if follow_info[1].text else '0'
            
            return {
                'uid': uid,
                'screen_name': nickname,
                'followers_count': followers,
                'follow_count': followings
            }
        except Exception as e:
            print(f"获取用户 {uid} 信息失败: {e}")
            return None
    
    def get_followings(self, uid: str, max_count: int = 100) -> list:
        """获取关注列表"""
        followings = []
        url = f'https://m.weibo.cn/api/container/getIndex?containerid=231051_-_followers_-_{uid}'
        
        self.driver.get(url)
        time.sleep(2)
        
        try:
            # 获取JSON响应
            pre = self.driver.find_element(By.TAG_NAME, 'pre')
            data = json.loads(pre.text)
            
            if data.get('ok') == 1:
                cards = data.get('data', {}).get('cards', [])
                for card in cards:
                    for item in card.get('card_group', []):
                        user = item.get('user', {})
                        if 'id' in user:
                            followings.append(str(user['id']))
                            if len(followings) >= max_count:
                                break
        except Exception as e:
            print(f"获取关注列表失败: {e}")
        
        return followings[:max_count]
    
    def close(self):
        """关闭浏览器"""
        self.driver.quit()


def main():
    """使用示例"""
    print("Selenium微博爬虫")
    print("注意: 需要安装 selenium 和 chrome driver")
    print("安装: pip install selenium")
    print("ChromeDriver: https://chromedriver.chromium.org/")


if __name__ == "__main__":
    main()

