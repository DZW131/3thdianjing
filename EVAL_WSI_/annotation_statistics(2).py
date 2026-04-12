import json
import numpy as np
import cv2
import os
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

def calculate_polygon_area(x_coords, y_coords):
    """
    使用鞋带公式计算多边形面积（像素数量）
    Args:
        x_coords: x坐标列表
        y_coords: y坐标列表
    Returns:
        int: 多边形面积（像素数量）
    """
    if len(x_coords) < 3 or len(y_coords) < 3:
        return 0
    
    # 确保多边形是闭合的
    if x_coords[0] != x_coords[-1] or y_coords[0] != y_coords[-1]:
        x_coords = x_coords + [x_coords[0]]
        y_coords = y_coords + [y_coords[0]]
    
    # 鞋带公式计算面积
    area = 0
    n = len(x_coords)
    for i in range(n - 1):
        area += x_coords[i] * y_coords[i + 1] - x_coords[i + 1] * y_coords[i]
    
    return abs(area) // 2

def getFileList(dir, Filelist, ext=None):
    """递归获取文件列表"""
    if not os.path.exists(dir):
        print(f"警告: 路径不存在: {dir}")
        return Filelist
    
    if os.path.isfile(dir):
        print(f"检查文件: {dir}")
        if ext is None:
            Filelist.append(dir)
            print(f"添加文件: {dir}")
        else:
            if dir.lower().endswith(ext.lower()):
                Filelist.append(dir)
                print(f"添加{ext}文件: {dir}")
    elif os.path.isdir(dir):
        print(f"搜索目录: {dir}")
        try:
            files = os.listdir(dir)
            print(f"目录中的文件: {files}")
            for s in files:
                newDir = os.path.join(dir, s)
                getFileList(newDir, Filelist, ext)
        except PermissionError:
            print(f"权限错误: 无法访问目录 {dir}")
        except Exception as e:
            print(f"错误: 访问目录 {dir} 时出错: {e}")
    return Filelist

def validate_json_file(json_path):
    """
    验证JSON文件是否有效
    Args:
        json_path: JSON文件路径
    Returns:
        bool: 文件是否有效
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查必要的字段
        if 'annotation' not in data:
            return False
        
        # 检查是否有有效的标注
        annotations = data['annotation']
        if not isinstance(annotations, list) or len(annotations) == 0:
            return False
        
        # 检查标注格式
        for annotation in annotations:
            if 'position' not in annotation or 'label' not in annotation:
                return False
            
            position = annotation['position']
            if 'x' not in position or 'y' not in position:
                return False
            
            # 检查坐标是否为有效列表
            if not isinstance(position['x'], list) or not isinstance(position['y'], list):
                return False
            
            if len(position['x']) != len(position['y']) or len(position['x']) < 3:
                return False
        
        return True
    except:
        return False

def read_json_annotation(json_path):
    """
    读取JSON标注文件
    Args:
        json_path: JSON文件路径
    Returns:
        list: 标注信息列表 [(x_coords, y_coords, color), ...]
    """
    annotations = []
    
    try:
        f = open(json_path, 'r')
        info_dict = json.load(f)
    except:
        try:
            f = open(json_path, 'r', encoding='gbk')
            info_dict = json.load(f)
        except:
            try:
                f = open(json_path, 'r', encoding='GBK')
                info_dict = json.load(f)
            except:
                try:
                    f = open(json_path, 'r', encoding='utf-8')
                    info_dict = json.load(f)
                except:
                    print(f"无法读取文件: {json_path}")
                    return annotations
    
    roilist = info_dict['annotation']
    
    for roi_dict in roilist:
        try:
            position = roi_dict["position"]
            color = roi_dict["label"]
            
            x_coords = position["x"]
            y_coords = position["y"]
            
            if len(x_coords) >= 3 and len(y_coords) >= 3:
                annotations.append((x_coords, y_coords, color))
        except:
            continue
    
    f.close()
    return annotations

def statistics_annotations(img_folder, output_file=None):
    """
    统计标注信息
    Args:
        img_folder: 包含JSON文件的文件夹路径
        output_file: 输出统计结果的文件路径
    """
    # 颜色到类别的映射
    color_to_label = {
        '#FF0000': 1,  # 严重损伤线粒体
        '#3DC005': 2,  # 中度损伤线粒体
        '#01F8E7': 3,  # 健康线粒体
        '#00FFB7': 4,  # 自噬线粒体
        '#E1FF00': 5,  # 肌浆网
        '#9D00FF': 6,  # 闰盘
        '#EE00FF': 7,  # Z线样物质堆积
        '#FFCC00': 8,  # 脂滴
        '#FF7700': 9,  # 糖原颗粒
        '#00AAFF': 10, # Z线
        '#14699D': 11, # T管
        '#84FF00': 12, # M线
        '#929C1E': 13  # 心肌侧管
    }
    
    # 类别名称映射
    label_to_name = {
        1: "严重损伤线粒体",
        2: "中度损伤线粒体", 
        3: "健康线粒体",
        4: "自噬线粒体",
        5: "肌浆网",
        6: "闰盘",
        7: "Z线样物质堆积",
        8: "脂滴",
        9: "糖原颗粒",
        10: "Z线",
        11: "T管",
        12: "M线",
        13: "心肌侧管"
    }
    
    # 统计数据
    class_pixel_count = defaultdict(int)  # 每个类别的像素数量
    class_region_count = defaultdict(int)  # 每个类别的区块数量
    file_statistics = {}  # 每个文件的统计信息
    unknown_colors = set()  # 未知颜色
    
    # 找到所有JSON文件
    json_files = []
    getFileList(img_folder, json_files, '.json')
    print(f"找到 {len(json_files)} 个JSON文件")
    
    valid_files = 0
    invalid_files = 0
    
    for json_path in tqdm(json_files, desc="统计标注信息"):
        file_name = os.path.basename(json_path)
        
        # 验证文件有效性
        if not validate_json_file(json_path):
            invalid_files += 1
            continue
        
        valid_files += 1
        
        # 读取标注信息
        annotations = read_json_annotation(json_path)
        
        if not annotations:
            continue
        
        # 统计当前文件的信息
        file_class_pixel = defaultdict(int)
        file_class_region = defaultdict(int)
        
        for x_coords, y_coords, color in annotations:
            if color not in color_to_label:
                unknown_colors.add(color)
                continue
            
            label = color_to_label[color]
            
            # 计算像素数量（多边形面积）
            pixel_count = calculate_polygon_area(x_coords, y_coords)
            
            # 更新全局统计
            class_pixel_count[label] += pixel_count
            class_region_count[label] += 1
            
            # 更新文件统计
            file_class_pixel[label] += pixel_count
            file_class_region[label] += 1
        
        # 保存文件统计信息
        file_statistics[file_name] = {
            'pixel_count': dict(file_class_pixel),
            'region_count': dict(file_class_region),
            'total_regions': sum(file_class_region.values()),
            'total_pixels': sum(file_class_pixel.values())
        }
    
    # 打印统计结果
    print(f"\n============ 标注统计结果 ============")
    print(f"总文件数: {len(json_files)}")
    print(f"有效文件数: {valid_files}")
    print(f"无效文件数: {invalid_files}")
    
    if unknown_colors:
        print(f"\n发现未知颜色: {unknown_colors}")
    
    print(f"\n============ 各类别统计 ============")
    print(f"{'类别ID':<4} {'类别名称':<15} {'区块数量':<10} {'像素数量':<15} {'平均像素/区块':<15}")
    print("-" * 70)
    
    total_regions = sum(class_region_count.values())
    total_pixels = sum(class_pixel_count.values())
    
    # 按类别ID排序
    for label in sorted(label_to_name.keys()):
        if label in class_region_count:
            name = label_to_name[label]
            regions = class_region_count[label]
            pixels = class_pixel_count[label]
            avg_pixels = pixels / regions if regions > 0 else 0
            
            print(f"{label:<4} {name:<15} {regions:<10} {pixels:<15} {avg_pixels:<15.2f}")
    
    print("-" * 70)
    print(f"{'总计':<4} {'':<15} {total_regions:<10} {total_pixels:<15} {total_pixels/total_regions if total_regions > 0 else 0:<15.2f}")
    
    # 计算比例
    print(f"\n============ 各类别比例 ============")
    print(f"{'类别ID':<4} {'类别名称':<15} {'区块比例(%)':<12} {'像素比例(%)':<12}")
    print("-" * 50)
    
    for label in sorted(label_to_name.keys()):
        if label in class_region_count:
            name = label_to_name[label]
            region_ratio = (class_region_count[label] / total_regions * 100) if total_regions > 0 else 0
            pixel_ratio = (class_pixel_count[label] / total_pixels * 100) if total_pixels > 0 else 0
            
            print(f"{label:<4} {name:<15} {region_ratio:<12.2f} {pixel_ratio:<12.2f}")
    
    # 保存详细统计到文件
    if output_file:
        save_detailed_statistics(file_statistics, class_pixel_count, class_region_count, 
                                label_to_name, output_file)
        print(f"\n详细统计结果已保存到: {output_file}")
    
    return {
        'class_pixel_count': dict(class_pixel_count),
        'class_region_count': dict(class_region_count),
        'file_statistics': file_statistics,
        'label_to_name': label_to_name,
        'unknown_colors': unknown_colors
    }

def save_detailed_statistics(file_statistics, class_pixel_count, class_region_count, 
                           label_to_name, output_file):
    """
    保存详细统计结果到文件
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("============ 标注统计详细报告 ============\n\n")
        
        # 总体统计
        f.write("总体统计:\n")
        f.write(f"总文件数: {len(file_statistics)}\n")
        f.write(f"总区块数: {sum(class_region_count.values())}\n")
        f.write(f"总像素数: {sum(class_pixel_count.values())}\n\n")
        
        # 各类别统计
        f.write("各类别统计:\n")
        f.write(f"{'类别ID':<4} {'类别名称':<15} {'区块数量':<10} {'像素数量':<15} {'平均像素/区块':<15}\n")
        f.write("-" * 70 + "\n")
        
        for label in sorted(label_to_name.keys()):
            if label in class_region_count:
                name = label_to_name[label]
                regions = class_region_count[label]
                pixels = class_pixel_count[label]
                avg_pixels = pixels / regions if regions > 0 else 0
                
                f.write(f"{label:<4} {name:<15} {regions:<10} {pixels:<15} {avg_pixels:<15.2f}\n")
        
        # 各文件统计
        f.write("\n\n各文件详细统计:\n")
        f.write("=" * 50 + "\n")
        
        for file_name, stats in file_statistics.items():
            f.write(f"\n文件: {file_name}\n")
            f.write(f"总区块数: {stats['total_regions']}\n")
            f.write(f"总像素数: {stats['total_pixels']}\n")
            f.write("类别分布:\n")
            
            for label, count in stats['region_count'].items():
                name = label_to_name[label]
                pixels = stats['pixel_count'][label]
                f.write(f"  {name}: {count}个区块, {pixels}像素\n")
            
            f.write("-" * 30 + "\n")

def plot_statistics(statistics_data, save_path=None):
    """
    绘制统计图表
    Args:
        statistics_data: 统计数据
        save_path: 保存图表的路径
    """
    class_pixel_count = statistics_data['class_pixel_count']
    class_region_count = statistics_data['class_region_count']
    label_to_name = statistics_data['label_to_name']
    
    # 准备数据
    labels = []
    region_counts = []
    pixel_counts = []
    
    for label in sorted(label_to_name.keys()):
        if label in class_region_count:
            labels.append(f"{label}:{label_to_name[label][:6]}")
            region_counts.append(class_region_count[label])
            pixel_counts.append(class_pixel_count[label])
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 区块数量柱状图
    ax1.bar(range(len(labels)), region_counts, color='skyblue')
    ax1.set_xlabel('类别')
    ax1.set_ylabel('区块数量')
    ax1.set_title('各类别区块数量统计')
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    
    # 在柱子上显示数值
    for i, v in enumerate(region_counts):
        ax1.text(i, v + max(region_counts) * 0.01, str(v), ha='center', va='bottom')
    
    # 像素数量柱状图
    ax2.bar(range(len(labels)), pixel_counts, color='lightcoral')
    ax2.set_xlabel('类别')
    ax2.set_ylabel('像素数量')
    ax2.set_title('各类别像素数量统计')
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    
    # 在柱子上显示数值
    for i, v in enumerate(pixel_counts):
        ax2.text(i, v + max(pixel_counts) * 0.01, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"统计图表已保存到: {save_path}")
    
    plt.show()

if __name__ == '__main__':
    # 设置参数 - 请根据您的实际路径修改
    img_folder = r'D:\msqtry\region_level_train\jijie json'  # JSON文件所在的文件夹
    output_file = r'D:\msqtry\region_level_train\jijie json/annotation_statistics.txt'  # 输出统计文件
    plot_save_path = r'D:\msqtry\region_level_train\jijie json\annotation_statistics.png'  # 图表保存路径
    
    # 执行统计
    print("开始统计标注信息...")
    statistics_data = statistics_annotations(img_folder, output_file)
    
    # 绘制统计图表
    print("绘制统计图表...")
    plot_statistics(statistics_data, plot_save_path)
    
    print("统计完成！")
