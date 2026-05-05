import os
import re

CSS_ADDITIONS = """
/* Profile Section Mockup Match */
.sidebar-profile-section {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding-bottom: 30px;
    margin-bottom: 20px;
    border-bottom: 1px solid var(--border-light);
}

.sidebar-avatar-wrapper {
    width: 95px;
    height: 95px;
    border-radius: 50%;
    border: 3px solid #ffffff;
    padding: 2px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: transparent;
}

.sidebar-avatar-wrapper img {
    width: 100%;
    height: 100%;
    border-radius: 50%;
    object-fit: cover;
    background: #e2e8f0;
}

.placeholder-avatar-xl {
    width: 100%;
    height: 100%;
    border-radius: 50%;
    background: rgba(255,255,255,0.1);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 38px;
    color: #ffffff;
}

.sidebar-name {
    font-size: 16px;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: 1px;
    margin-bottom: 4px;
    word-break: break-all;
    text-align: center;
}

.sidebar-role {
    font-size: 11px;
    color: #8ba1b5;
    letter-spacing: 1.5px;
    font-weight: 600;
}

/* Nav Item Arrow Highlight Effect */
.sidebar-nav {
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.nav-item {
    position: relative;
    padding: 16px 20px;
    border-radius: 0;
    color: #8ba1b5;
    font-weight: 600;
    font-size: 15px;
    text-decoration: none;
    transition: all 0.3s ease;
    display: flex;
    align-items: center;
    gap: 20px;
    cursor: pointer;
    border: none;
    background: transparent;
}

.nav-item:hover {
    color: #ffffff;
}

.nav-item.active {
    background: #3eb4fa;
    color: #ffffff;
    margin-left: -20px;
    margin-right: -20px;
    padding-left: 40px;
    box-shadow: none;
}

.nav-item.active::after {
    content: '';
    position: absolute;
    right: -12px;
    top: 50%;
    transform: translateY(-50%);
    border-width: 12px 0 12px 12px;
    border-style: solid;
    border-color: transparent transparent transparent #3eb4fa;
    width: 0;
    height: 0;
    z-index: 10;
}

.main-content {
    background: #f1f3f6; /* Lighter background like mockup */
}
"""

SIDEBAR_PROFILE_HTML = """
            <div class="sidebar-profile-section">
                <div class="sidebar-avatar-wrapper">
                    {% if current_doctor and current_doctor.profile_pic %}
                    <img src="{{ url_for('static', filename=current_doctor.profile_pic) }}" alt="Profile">
                    {% else %}
                    <div class="placeholder-avatar-xl">👤</div>
                    {% endif %}
                </div>
                <div class="sidebar-name">{{ current_doctor.username.upper() if current_doctor else 'UNKNOWN DOCTOR' }}</div>
                <div class="sidebar-role">LOREM IPSUM</div>
            </div>
"""

ACCOUNT_LINK_HTML = """
                <a href="{{ url_for('account') }}" class="nav-item {% if request.endpoint == 'account' %}active{% endif %}" style="margin-top: 30px;">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>
                    Setting
                </a>
"""

def update_templates():
    files = ['templates/index.html', 'templates/result.html', 'templates/account.html']
    
    for f in files:
        with open(f, 'r', encoding='utf-8') as file:
            content = file.read()
            
        # 1. Remove <header class="top-header"> block completely
        content = re.sub(r'<header class="top-header">.*?</header>', '', content, flags=re.DOTALL)
        
        # 2. Add SIDEBAR_PROFILE_HTML after <div class="sidebar-logo">...</div>
        if 'sidebar-profile-section' not in content:
            content = re.sub(r'(<div class="sidebar-logo">.*?</div>)', r'\1\n' + SIDEBAR_PROFILE_HTML, content, flags=re.DOTALL)
            
        # 3. Inject Setting link before </nav> if not exists
        if 'Setting' not in content:
            content = content.replace('</nav>', ACCOUNT_LINK_HTML + '\n            </nav>')
            
        with open(f, 'w', encoding='utf-8') as file:
            file.write(content)

def update_css():
    with open('static/css/style.css', 'a', encoding='utf-8') as file:
        file.write(CSS_ADDITIONS)
        
        
if __name__ == '__main__':
    update_templates()
    update_css()
    print("Files Updated Successfully")
