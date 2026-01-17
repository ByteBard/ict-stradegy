"""
PDF切割工具 - 将大PDF文件按页数切割成小文件
用于处理阿布价格行为学PDF
"""

import os
from pathlib import Path
from PyPDF2 import PdfReader, PdfWriter


def get_pdf_info(pdf_path: str) -> dict:
    """获取PDF基本信息"""
    reader = PdfReader(pdf_path)
    file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # MB
    return {
        "path": pdf_path,
        "pages": len(reader.pages),
        "size_mb": round(file_size, 2)
    }


def split_pdf(pdf_path: str, output_dir: str, pages_per_file: int = 5) -> list[str]:
    """
    按页数切割PDF

    Args:
        pdf_path: 源PDF路径
        output_dir: 输出目录
        pages_per_file: 每个文件包含的页数

    Returns:
        生成的文件路径列表
    """
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 获取原文件名(不含扩展名)
    base_name = Path(pdf_path).stem

    output_files = []

    for start_page in range(0, total_pages, pages_per_file):
        end_page = min(start_page + pages_per_file, total_pages)

        writer = PdfWriter()
        for page_num in range(start_page, end_page):
            writer.add_page(reader.pages[page_num])

        # 输出文件名: 原名_页码范围.pdf
        output_name = f"{base_name}_p{start_page+1:03d}-{end_page:03d}.pdf"
        output_path = os.path.join(output_dir, output_name)

        with open(output_path, "wb") as f:
            writer.write(f)

        output_files.append(output_path)
        print(f"已生成: {output_name} (第{start_page+1}-{end_page}页)")

    return output_files


def main():
    # PDF文件路径
    pdfs = [
        r"C:\Users\HP\Downloads\阿布价格行为学（基础篇）_01-36.pdf",
        r"C:\Users\HP\Downloads\阿布价格行为学（进阶篇）_37-52.pdf"
    ]

    # 输出目录
    output_base = r"C:\Repo\ict-stradegy\docs\pdf_split"

    for pdf_path in pdfs:
        print(f"\n{'='*50}")
        print(f"处理: {Path(pdf_path).name}")

        # 获取PDF信息
        info = get_pdf_info(pdf_path)
        print(f"总页数: {info['pages']}, 大小: {info['size_mb']}MB")

        # 根据文件名确定输出子目录
        if "基础" in pdf_path:
            output_dir = os.path.join(output_base, "basic")
        else:
            output_dir = os.path.join(output_base, "advanced")

        # 切割PDF (每5页一个文件)
        files = split_pdf(pdf_path, output_dir, pages_per_file=5)
        print(f"共生成 {len(files)} 个文件")


if __name__ == "__main__":
    main()
