# 🔧 Live Server 兼容性修复

## 问题说明

Live Server会自动监测文件变化并刷新页面，这会导致演示进度丢失。

## ✅ 已修复

HTML文件已更新，添加了以下功能：

### 1. 自动检测Live Server环境
- 当在 `localhost` 或 `127.0.0.1` 上运行时，自动启用状态保存
- 页面右上角会显示黄色提示框

### 2. 状态自动保存
- 当前阶段（身份去匿名化/属性推断/鲁棒性/防御）
- 当前选择的方法
- 当前演示步骤
- 保存在浏览器的 `localStorage`，5分钟有效

### 3. 刷新后自动恢复
- Live Server刷新后，自动恢复到之前的状态
- 包括阶段选择、方法选择、演示进度

## 🚀 使用方法

### 方法1: 不使用Live Server（推荐）

**直接用浏览器打开HTML文件，完全避免自动刷新：**

```bash
# 在终端运行（不需要root权限）
xdg-open /home/honglianglu/hdd/Anonymous/results/attack_demo_improved.html
```

或者在文件管理器中**双击HTML文件**。

### 方法2: 继续使用Live Server（已修复）

现在即使Live Server自动刷新，你的演示进度也会保存：

1. 用Live Server打开HTML
2. 选择阶段和方法
3. 开始演示
4. **即使Live Server刷新，也会自动恢复到当前进度** ✅

## 📱 如何验证修复

1. 用Live Server打开 `attack_demo_improved.html`
2. 你会在右上角看到黄色提示框：
   ```
   💡 提示
   检测到Live Server。建议直接用浏览器打开HTML文件以避免自动刷新。
   点击关闭 · 已启用状态保存
   ```
3. 选择一个阶段和方法
4. 点击几次"下一步"
5. 刷新页面（F5）
6. **你会发现演示自动恢复到刷新前的状态** ✅

## 💡 最佳实践

### 推荐：直接用浏览器打开

```bash
# 方法1: 使用xdg-open（最简单）
xdg-open /home/honglianglu/hdd/Anonymous/results/attack_demo_improved.html

# 方法2: 如果xdg-open不可用，手动指定浏览器
# 查看可用的浏览器
ls /usr/bin/ | grep -E 'firefox|chrome|chromium'

# 使用找到的浏览器
/usr/bin/firefox /home/honglianglu/hdd/Anonymous/results/attack_demo_improved.html
# 或
/usr/bin/chromium-browser /home/honglianglu/hdd/Anonymous/results/attack_demo_improved.html

# 方法3: 在文件管理器中找到HTML文件，双击打开
```

### 如果必须用Live Server

现在状态会自动保存，但还是建议：
1. 完成一个完整的演示后再切换方法
2. 避免在演示过程中编辑HTML文件（会触发刷新）
3. 重置演示前先完成当前演示

## 🔍 技术细节

### 保存的状态
```javascript
{
    phase: "deanonymization",     // 当前阶段
    methodId: "greedy",            // 当前方法ID
    step: 3,                       // 当前步骤（第3步）
    timestamp: 1704067200000       // 保存时间戳
}
```

### 状态有效期
- 5分钟内有效
- 超过5分钟自动清除，避免混淆

### 清除状态
- 点击"重置"按钮会清除保存的状态
- 关闭浏览器标签页会保留状态（5分钟内）
- 超过5分钟自动过期

## 🎯 对比

| 场景 | 修复前 | 修复后 |
|------|--------|--------|
| Live Server刷新 | ❌ 演示重置 | ✅ 自动恢复 |
| 切换阶段 | ✅ 正常 | ✅ 正常+保存 |
| 点击下一步 | ✅ 正常 | ✅ 正常+保存 |
| 点击重置 | ✅ 正常 | ✅ 正常+清除保存 |
| 直接用浏览器 | ✅ 完美 | ✅ 完美（无刷新） |

## 📝 快速测试

```bash
# 1. 确保文件已更新
ls -lh /home/honglianglu/hdd/Anonymous/results/attack_demo_improved.html

# 2. 用xdg-open打开（推荐）
xdg-open /home/honglianglu/hdd/Anonymous/results/attack_demo_improved.html

# 3. 或者找到你的浏览器路径
which firefox
which chromium-browser
which google-chrome

# 4. 然后用浏览器打开
$(which firefox) /home/honglianglu/hdd/Anonymous/results/attack_demo_improved.html &
```

## ⚠️ 注意事项

1. **状态保存仅在Live Server环境下启用**
   - 直接用浏览器打开时不需要状态保存（因为不会刷新）
   
2. **不要在多个标签页同时打开**
   - localStorage是共享的，多个标签页会互相覆盖状态

3. **隐私模式/无痕模式**
   - localStorage可能不可用，状态保存会失败
   - 但不影响正常使用，只是刷新后需要重新选择

## 🎉 总结

**最简单的解决方案：**
```bash
xdg-open /home/honglianglu/hdd/Anonymous/results/attack_demo_improved.html
```

这样完全避免Live Server的自动刷新问题，不需要任何权限。

**如果必须用Live Server：**
- 已添加状态保存功能 ✅
- 刷新后自动恢复 ✅
- 但还是建议直接用浏览器打开更流畅

---

**问题解决了吗？试试 `xdg-open` 命令！** 🚀









