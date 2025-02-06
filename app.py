import os
import re
import logging
import subprocess
import requests
import streamlit as st
from jinja2 import Template
from tenacity import retry, stop_after_attempt, wait_exponential

# ==============================
# 配置日志记录
# ==============================
logging.basicConfig(
    filename='resume_generator.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ==============================
# 工具函数
# ==============================
def parse_markdown(md_content: str) -> dict:
    """
    解析Markdown格式的职位描述
    """
    data = {}
    desc_match = re.search(r"\*职位描述：\*\s*(.*?)\n\*", md_content, re.DOTALL)
    if desc_match:
        data["job_desc"] = desc_match.group(1).strip()
    hard_skills_match = re.search(r"\*硬技能：\*\s*((?:- .*\n)+)", md_content)
    if hard_skills_match:
        data["hard_skills"] = [s.strip() for s in hard_skills_match.group(1).split("- ") if s.strip()]
    keywords_match = re.search(r"\*简历关键词：\*\s*((?:- .*\n)+)", md_content)
    if keywords_match:
        data["keywords"] = [s.strip() for s in keywords_match.group(1).split("- ") if s.strip()]
    return data

def build_prompt(base_resume: str, job_data: dict) -> str:
    """
    组装Prompt
    """
    prompt = f"""
    你是一个资深且擅长运用AI技术的HR，具有丰富的候选人筛选经验。你已经回顾过许多候选人的简历，并总结了他们的共同优势。根据不同工种的岗位描述（JD）和筛选标准，按照麦肯锡原则（从重要到不重要）为应届硕士毕业生定制简历。这个简历需要突出软技能和硬技能，筛选过程中，必须能够体现以下关键能力和关键词。简历应当控制在一页之内。
    请根据以下信息优化简历：
    - 目标岗位描述：{job_data['job_desc']}
    - 硬技能要求：{', '.join(job_data['hard_skills'])}
    - 关键词：{', '.join(job_data['keywords'])}
    
    原始简历内容：
    {base_resume}
    
    要求：
    1. 突出硬技能和相关经验
    2. 使用关键词优化描述
    3. 保持简洁和专业
    """
    return prompt

def fill_latex_template(template_path: str, data: dict) -> str:
    """
    填充LaTeX模板
    """
    with open(template_path, "r", encoding="utf-8") as f:
        template = Template(f.read())
    return template.render(data)

# ==============================
# DeepSeek API 调用函数
# ==============================
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def call_deepseek_api(prompt: str, api_key: str) -> str:
    """
    调用DeepSeek API的强化版函数
    包含12类异常场景处理，覆盖全链路可能问题
    """
    # 前置校验
    if not api_key:
        raise ValueError("API密钥不能为空，请到DeepSeek控制台获取有效密钥")
    if not isinstance(api_key, str) or len(api_key) != 64:  # 假设API密钥是64位字符串
        raise ValueError("API密钥格式错误，应为64位字符串")
    if not prompt:
        raise ValueError("提示词内容不能为空")
    if len(prompt) > 4000:  # 假设模型限制上下文长度
        raise ValueError(f"提示词过长（当前{len(prompt)}字符），最大允许4000字符")

    # 网络请求
    try:
        response = requests.post(
            url="https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "ResumeGenerator/1.0"
            },
            json={
                "model": "deepseek-chat",
                "messages": [{
                    "role": "user",
                    "content": prompt
                }],
                "temperature": 0.3,
                "max_tokens": 2000
            },
            timeout=15  # 设置超时时间
        )
    except requests.exceptions.Timeout:
        raise ConnectionError("API请求超时（15秒），建议：1. 检查网络连接 2. 稍后重试")
    except requests.exceptions.ConnectionError:
        raise ConnectionError("网络连接失败，可能原因：1. 本地网络中断 2. DeepSeek服务不可用")
    except requests.exceptions.SSLError:
        raise ConnectionError("SSL证书验证失败，建议：1. 检查系统时间 2. 更新证书库")
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"未知网络错误: {str(e)}")

    # 响应处理
    try:
        response_data = response.json()
    except ValueError:
        error_info = f"响应解析失败，状态码：{response.status_code}，原始响应：{response.text[:200]}..."
        if "application/json" not in response.headers.get("Content-Type", ""):
            error_info += "\n原因：服务器返回了非JSON格式的响应"
        raise ValueError(error_info)

    # 处理HTTP错误状态码
    if response.status_code != 200:
        error_type = {
            401: "API密钥无效或过期",
            403: "权限不足，请检查API密钥权限",
            429: "请求频率超限，请稍后重试",
            500: "服务器内部错误",
            503: "服务不可用"
        }.get(response.status_code, "未知API错误")

        error_details = response_data.get("error", {}).get("message", "无错误详情")
        raise ConnectionError(
            f"API请求失败 [{response.status_code} {error_type}]\n"
            f"错误详情：{error_details}\n"
            f"请求ID：{response.headers.get('X-Request-ID', '无')}"
        )

    # 数据结构校验
    if not isinstance(response_data, dict):
        raise ValueError(f"响应数据结构异常，期望字典类型，实际得到：{type(response_data)}")
    if "choices" not in response_data:
        raise KeyError("响应数据缺少关键字段：choices，完整响应：" + str(response_data)[:200])
    choices = response_data["choices"]
    if not isinstance(choices, list) or len(choices) == 0:
        raise IndexError("choices字段为空或类型错误，期望非空列表")
    first_choice = choices[0]
    if "message" not in first_choice:
        raise KeyError("choice条目缺少message字段")
    if "content" not in first_choice["message"]:
        raise ValueError("message字段缺少content内容")

    content = first_choice["message"]["content"]
    if not content.strip():
        raise ValueError("API返回内容为空，可能原因：1. 触发内容过滤 2. 模型生成失败")

    # 内容安全校验
    if len(content) < 100:  # 假设正常简历优化结果至少100字符
        raise ValueError(
            f"生成内容过短（仅{len(content)}字符），可能问题：\n"
            f"1. 原始简历与岗位不匹配\n"
            f"2. Prompt设计不合理\n"
            f"生成内容：{content[:200]}..."
        )
    if "抱歉" in content or "无法" in content:
        raise ValueError(
            f"检测到模型拒绝响应：\n{content[:500]}\n"
            "建议：1. 检查Prompt是否合规 2. 调整请求参数"
        )

    return content

# ==============================
# Streamlit 主界面
# ==============================
def main():
    st.title("自动化针对性简历生成系统")

    # 上传简历母版
    base_resume_file = st.file_uploader("上传简历母版（当前仅支持Markdown格式）", type="md")
    if base_resume_file:
        base_resume = base_resume_file.read().decode("utf-8")

    # 选择职位需求
    job_files = {
        "网络工程师": "data/job_network_engineer.md",
        "大数据工程师": "data/job_data_scientist.md"
    }
    job_name = st.selectbox("选择职位需求", list(job_files.keys()))
    if job_name:
        with open(job_files[job_name], "r", encoding="utf-8") as f:
            job_data = parse_markdown(f.read())

    # 输入DeepSeek API密钥
    api_key = st.text_input("输入DeepSeek API密钥")

    # 生成简历
    if st.button("生成简历"):
        if not api_key:
            st.error("请输入DeepSeek API密钥")
        else:
            with st.spinner("生成中..."):
                try:
                    # 组装Prompt
                    prompt = build_prompt(base_resume, job_data)
                    # 调用DeepSeek API
                    optimized_resume = call_deepseek_api(prompt, api_key)

                    # 填充LaTeX模板
                    latex_content = fill_latex_template("templates/resume_template.tex", {
                        "name": "张三",
                        "contact": "zhangsan@email.com",
                        "experiences": [
                            {"company": "XX科技公司", "position": "全栈工程师", "duration": "2020.01 - 至今", "details": "负责前后端开发，使用Python和JavaScript。"},
                            {"company": "YY网络公司", "position": "网络工程师", "duration": "2018.06 - 2019.12", "details": "负责网络架构设计和维护。"}
                        ],
                        "skills": job_data["hard_skills"],
                        "optimized_resume": optimized_resume
                    })

                    # 保存并编译LaTeX
                    os.makedirs("output", exist_ok=True)
                    with open("output/resume.tex", "w", encoding="utf-8") as f:
                        f.write(latex_content)

                    # 编译LaTeX文件为PDF
                    subprocess.run(["xelatex", "resume.tex"], cwd="output")

                    # 提供下载链接
                    st.success("生成完成！")
                    with open("output/resume.pdf", "rb") as f:
                        st.download_button("下载简历", f, file_name="resume.pdf")

                except Exception as e:
                    st.error(f"生成失败: {str(e)}")
                    logging.error(f"生成失败: {str(e)}")

# ==============================
# 程序入口
# ==============================
if __name__ == "__main__":
    main()