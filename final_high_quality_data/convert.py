import json
import os

def convert_json_to_jsonl(input_path):
    """
    将用户指定的 JSON 文件转换为同名 JSONL 文件（输出到同目录）
    
    参数:
        input_path (str): 用户指定的输入文件路径（必须为 .json 文件）
    """
    # 检查输入文件是否存在
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    # 生成输出路径（相同目录、同名、后缀为 .jsonl）
    dir_name = os.path.dirname(input_path)
    base_name = os.path.basename(input_path)
    output_name = os.path.splitext(base_name)[0] + ".jsonl"
    output_path = os.path.join(dir_name, output_name)
    
    # 读取并转换数据
    with open(input_path, 'r', encoding='utf-8') as f_in:
        try:
            data = json.load(f_in)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 解析失败: {e}")
    
    # 写入 JSONL 文件
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for item in data:
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"转换完成！输出文件已保存至: {output_path}")

# 使用示例
if __name__ == "__main__":
    # input_file = input("请输入 JSON 文件路径（例如：./data/input.json）: ").strip()
    convert_json_to_jsonl('./final_high_quality_data/result_20250512_1109.json')