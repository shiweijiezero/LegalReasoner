import os
from typing import List, Tuple


def find_en_directories(base_path: str) -> List[Tuple[str, str]]:
    """
    Find all 'EN' directories and create corresponding output paths.
    Returns list of tuples (input_dir, output_dir).
    """
    input_output_pairs = []

    for root, dirs, _ in os.walk(base_path):
        if 'EN' in dirs:
            input_dir = os.path.join(root, 'EN')
            relative_path = os.path.relpath(input_dir, base_path)
            output_dir = os.path.join('processed_data', relative_path)
            input_output_pairs.append((input_dir, output_dir))

    return input_output_pairs


def rename_files_with_spaces(input_output_pairs: List[Tuple[str, str]]) -> None:
    """
    Rename all files containing spaces to use underscores in their names.

    Args:
        input_output_pairs: List of tuples containing (input_dir, output_dir) paths
    """
    renamed_files = []

    for input_dir, _ in input_output_pairs:
        try:
            # 遍历目录中的所有文件
            for root, dirs, files in os.walk(input_dir):
                # 首先处理文件名
                for file in files:
                    if ' ' in file:
                        old_path = os.path.join(root, file)
                        new_name = file.replace(' ', '_')
                        new_path = os.path.join(root, new_name)

                        try:
                            os.rename(old_path, new_path)
                            renamed_files.append((old_path, new_path))
                            print(f"已重命名: {old_path} -> {new_path}")
                        except OSError as e:
                            print(f"重命名文件时出错 {old_path}: {e}")

                # 然后处理目录名
                for dir_name in dirs:
                    if ' ' in dir_name:
                        old_dir_path = os.path.join(root, dir_name)
                        new_dir_name = dir_name.replace(' ', '_')
                        new_dir_path = os.path.join(root, new_dir_name)

                        try:
                            os.rename(old_dir_path, new_dir_path)
                            renamed_files.append((old_dir_path, new_dir_path))
                            print(f"已重命名目录: {old_dir_path} -> {new_dir_path}")
                        except OSError as e:
                            print(f"重命名目录时出错 {old_dir_path}: {e}")
        except OSError as e:
            print(f"处理目录时出错 {input_dir}: {e}")


    # 打印总结信息
    if renamed_files:
        print(f"\n总共重命名了 {len(renamed_files)} 个文件/目录:")
        for old, new in renamed_files:
            print(f"  {old} -> {new}")
    else:
        print("没有找到需要重命名的文件或目录")


def main():
    base_path = "."
    input_output_pairs = find_en_directories(base_path)

    if not input_output_pairs:
        print("未找到任何EN目录")
        return

    print("找到以下EN目录:")
    for input_dir, output_dir in input_output_pairs:
        print(f"输入目录: {input_dir}")
        print(f"输出目录: {output_dir}")

    # 执行重命名操作
    rename_files_with_spaces(input_output_pairs)


if __name__ == "__main__":
    main()
