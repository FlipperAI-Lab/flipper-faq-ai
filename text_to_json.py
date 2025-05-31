import json
text = """
сюда текст для ответов
"""
print(json.dumps(text, ensure_ascii=False))