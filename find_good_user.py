"""
测试脚本：找一个有关注关系的微博用户
"""

import sys
sys.path.insert(0, '/home/honglianglu/hdd/deanony')

from crawlers.weibo_final import RobustWeiboCrawler

# 一些知名微博用户ID（通常有很多关注关系）
test_uids = [
    "1642591402",  # 人民日报
    "1638782947",  # 央视新闻
    "1784473157",  # 新浪科技
    "1197161814",  # 微博小秘书
    "5044281310",  # 小米公司
]

cookies = "SINAGLOBAL=6740185828856.008.1764257392979; XSRF-TOKEN=Myn4TmTnG35cSjgyYPIJfvmV; SCF=AjiMSHwPp3pk5eVrMx10d6WYKiUi8q5VEC2hifoXmNfxm-mQDE2IPwP4DaI7i_6W3iyQ4sat5D1N02_MdRCywNM.; SUB=_2A25EStb9DeRhGeBP41cR8y3NyDuIHXVnJlY1rDV8PUNbmtANLUbakW9NRTnmMHLzxa3KXAOJoUwYFxbbtUflUmvP; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WFUHoUkz0PekjTvoM.HlSOx5JpX5KzhUgL.Foqp1h-7e0epe0M2dJLoIEXLxKBLBo.L12eLxK.LB.-L1K.LxKnL12eLBoqLxKML1K-LB-2LxK.L1K-LB.qt; ALF=02_1769354157; _s_tentry=weibo.com; Apache=7516984194176.165.1766762537792; ULV=1766762537794:2:1:1:7516984194176.165.1766762537792:1764257392981; WBPSESS=2JbmQMfDBf9GhITJyWUUWznL60fHFOFR2V0qqV--Q6QQ6CjSe-HiZ0xa9TFn-LDSQqDlY1BaeyAkFOeWqX_zXuy2IQtbUl_bkq6V5XSWjW4mXeVHy0BlQrpgbFODloUw3x_fxkG6hoMnOUUDzYCjtA=="

crawler = RobustWeiboCrawler(cookies)

print("测试几个知名微博用户，找一个有关注关系的：\n")

for uid in test_uids:
    print(f"测试用户 {uid}...")
    user_info = crawler.get_user_info(uid)
    
    if user_info and user_info.get('screen_name'):
        print(f"  ✅ {user_info['screen_name']}")
        print(f"     粉丝: {user_info['followers_count']}")
        print(f"     关注: {user_info['follow_count']}")
        
        if user_info['follow_count'] > 0:
            print(f"     👉 这个用户可用！有 {user_info['follow_count']} 个关注")
            print(f"     推荐使用 UID: {uid}\n")
            break
    else:
        print(f"  ❌ 无法获取\n")
    
    import time
    time.sleep(2)

