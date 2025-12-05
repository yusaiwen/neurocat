document.addEventListener('DOMContentLoaded', function() {
    // 1. 创建切换按钮的 HTML
    var link_text = "Switch to Chinese";
    var target_lang = "zh";
    
    // 如果当前是中文环境（这一步需在中文版目录下反过来写）
    if (window.location.pathname.includes('/zh/')) {
        link_text = "切换为英文";
        target_lang = "en";
    }

    // 2. 插入到页面右上角或其他位置 (这里以 sphinx_rtd_theme 的面包屑导航栏为例)
    var navBar = document.querySelector('.wy-breadcrumbs-aside');
    if (navBar) {
        var btn = document.createElement('a');
        btn.innerHTML = link_text;
        // btn.className = "fa fa-language"; // 使用 font-awesome 图标
        btn.style.marginLeft = "10px";
        btn.style.cursor = "pointer";
        
        // 3. 点击事件：替换 URL 中的 /en/ 为 /zh/
        btn.onclick = function() {
            var currentPath = window.location.pathname;
            var newPath = "";
            
            if (target_lang === 'zh') {
                // 把 /en/ 替换为 /zh/
                newPath = currentPath.replace('/en/', '/zh/');
            } else {
                // 把 /zh/ 替换为 /en/
                newPath = currentPath.replace('/zh/', '/en/');
            }
            window.location.href = newPath;
        };
        
        navBar.appendChild(btn);
    }
});