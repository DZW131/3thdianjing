import sqlite3
import json
from pydaily import filesystem
import os
def db2json(db_path, json_path):

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    rois = []
    try:
        cursor.execute(
            "select method,groupName,position from Mark_label_her2Area left join MarkGroup on Mark_label_her2Area.groupId= MarkGroup.id")

    except:
        os.remove(db_path)
        return 0

    contents = cursor.fetchall()
    for content in contents:
        roi = {'method': content[0], 'remark': content[1], 'path': json.loads(content[2])}
        rois.append(roi)
    table = {'roilist': rois}
    json_str = json.dumps(table, ensure_ascii=False, indent=4)
    with open(json_path, 'w', encoding='utf-8') as writer:
        writer.write(json_str)


if __name__ == '__main__':
    src_db_path = '/media/zhouzhihao/DeepInformatic_dataset/lihansheng/HER2/免疫组化her2切片/7.29区域标注_db'
    save_dir='/media/zhouzhihao/DeepInformatic_dataset/lihansheng/HER2/免疫组化her2切片/7.29区域标注_json'
    db_files = filesystem.find_ext_files(src_db_path, ".db")
    os.makedirs(save_dir, exist_ok=True)
    for a_db_file in db_files:
        filepath = a_db_file.split(src_db_path)[1][1:]
        save_path = os.path.join(save_dir, filepath).replace('.db','.json')
        dirname = os.path.dirname(save_path)
        os.makedirs(dirname,exist_ok=True)
        db2json(a_db_file, save_path)
