import os
import json
import shutil


def backup_json_file(file_path, backup_dir):
    """
    创建 JSON 文件的备份。
    """
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    shutil.copy(file_path, backup_dir)
    print(f"备份创建: {file_path} -> {backup_dir}")


def replace_remark_with_label(json_dir, backup=True):
    """
    遍历指定目录内的所有 JSON 文件，将 'remark' 键替换为 'label'。

    参数:
    - json_dir: 包含 JSON 文件的目录路径。
    - backup: 是否创建备份（默认 True）。
    """
    # 定义备份目录
    backup_dir = os.path.join(json_dir, "backup_json_files")

    # 遍历目录及其子目录
    for root, dirs, files in os.walk(json_dir):
        for file in files:
            if file.lower().endswith('.json'):
                file_path = os.path.join(root, file)

                try:
                    # 如果需要备份，先创建备份
                    if backup:
                        backup_json_file(file_path, backup_dir)

                    # 读取 JSON 文件
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # 检查 'annotation' 键是否存在且为列表
                    if 'annotation' in data and isinstance(data['annotation'], list):
                        modified = False
                        for annotation in data['annotation']:
                            if 'strokeColor' in annotation:
                                annotation['label'] = annotation.pop('strokeColor')
                                modified = True

                        if modified:
                            # 将修改后的数据写回文件
                            with open(file_path, 'w', encoding='utf-8') as f:
                                json.dump(data, f, ensure_ascii=False, indent=4)
                            print(f"修改完成: {file_path}")
                        else:
                            print(f"无 'remark' 键需修改: {file_path}")
                    else:
                        print(f"文件格式不符合预期（缺少 'annotation' 键或不是列表）: {file_path}")

                except json.JSONDecodeError as e:
                    print(f"JSON 解析错误 ({file_path}): {e}")
                except Exception as e:
                    print(f"处理文件时出错 ({file_path}): {e}")


if __name__ == "__main__":
    # 指定包含 JSON 文件的目录路径
    json_directory = r'D:\msqtry\muscle indicator\json'

    # 调用函数进行替换
    replace_remark_with_label(json_directory, backup=True)