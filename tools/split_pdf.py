"""
PDF切割工具 - 将大PDF文件按页数切割成小文件
用于处理阿布价格行为学PDF

增强版：跳过损坏页面，继续处理
"""

import os
import sys
from pathlib import Path
from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.errors import PdfReadError
import warnings

# 忽略PyPDF2的警告
warnings.filterwarnings("ignore")


def get_pdf_info(pdf_path: str) -> dict:
    """获取PDF基本信息"""
    reader = PdfReader(pdf_path)
    file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # MB
    return {
        "path": pdf_path,
        "pages": len(reader.pages),
        "size_mb": round(file_size, 2)
    }


def extract_single_page(reader: PdfReader, page_num: int) -> tuple:
    """
    安全地提取单个页面

    Returns:
        (page_object, error_message) - 成功时error_message为None
    """
    try:
        page = reader.pages[page_num]
        # 尝试克隆页面来验证它是否可用
        writer = PdfWriter()
        writer.add_page(page)
        return page, None
    except Exception as e:
        return None, str(e)


def split_pdf_safe(pdf_path: str, output_dir: str, pages_per_file: int = 5) -> dict:
    """
    安全地按页数切割PDF，跳过损坏页面

    Returns:
        {
            "output_files": [...],
            "skipped_pages": [...],
            "total_processed": int,
            "total_skipped": int
        }
    """
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    base_name = Path(pdf_path).stem

    output_files = []
    skipped_pages = []

    for start_page in range(0, total_pages, pages_per_file):
        end_page = min(start_page + pages_per_file, total_pages)

        writer = PdfWriter()
        pages_added = 0
        batch_skipped = []

        for page_num in range(start_page, end_page):
            try:
                page = reader.pages[page_num]
                writer.add_page(page)
                pages_added += 1
            except Exception as e:
                skipped_pages.append({
                    "page": page_num + 1,
                    "error": str(e)[:100]
                })
                batch_skipped.append(page_num + 1)
                continue

        if pages_added > 0:
            # 生成文件名
            output_name = f"{base_name}_p{start_page+1:03d}-{end_page:03d}.pdf"
            output_path = os.path.join(output_dir, output_name)

            try:
                with open(output_path, "wb") as f:
                    writer.write(f)
                output_files.append(output_path)

                status = f"已生成: {output_name} ({pages_added}页)"
                if batch_skipped:
                    status += f" [跳过: {batch_skipped}]"
                print(status)
            except Exception as e:
                print(f"写入失败 {output_name}: {e}")
        else:
            print(f"跳过空批次: 第{start_page+1}-{end_page}页 (全部损坏)")

    return {
        "output_files": output_files,
        "skipped_pages": skipped_pages,
        "total_processed": len(output_files) * pages_per_file - len(skipped_pages),
        "total_skipped": len(skipped_pages)
    }


def split_pdf_by_text(pdf_path: str, output_dir: str, pages_per_file: int = 10) -> dict:
    """
    另一种方法：提取文本而不是复制页面
    适用于页面损坏但文本可读的情况
    """
    # 这是备用方案，如果需要可以实现
    pass


def main():
    # PDF文件路径
    pdfs = [
        r"C:\Users\HP\Downloads\阿布价格行为学（基础篇）_01-36.pdf",
        r"C:\Users\HP\Downloads\阿布价格行为学（进阶篇）_37-52.pdf"
    ]

    # 输出目录
    output_base = r"C:\Repo\ict-stradegy\docs\pdf_split"

    # 统计
    total_stats = {
        "files_created": 0,
        "pages_skipped": 0
    }

    for pdf_path in pdfs:
        if not os.path.exists(pdf_path):
            print(f"\n文件不存在: {pdf_path}")
            continue

        print(f"\n{'='*60}")
        print(f"处理: {Path(pdf_path).name}")

        try:
            info = get_pdf_info(pdf_path)
            print(f"总页数: {info['pages']}, 大小: {info['size_mb']}MB")
        except Exception as e:
            print(f"无法读取PDF信息: {e}")
            continue

        # 根据文件名确定输出子目录
        if "基础" in pdf_path:
            output_dir = os.path.join(output_base, "basic")
        else:
            output_dir = os.path.join(output_base, "advanced")

        # 安全切割PDF
        result = split_pdf_safe(pdf_path, output_dir, pages_per_file=5)

        print(f"\n统计:")
        print(f"  生成文件: {len(result['output_files'])}")
        print(f"  跳过页面: {result['total_skipped']}")

        if result['skipped_pages']:
            print(f"  损坏页面列表: {[p['page'] for p in result['skipped_pages'][:20]]}")
            if len(result['skipped_pages']) > 20:
                print(f"  ... 及其他 {len(result['skipped_pages']) - 20} 页")

        total_stats["files_created"] += len(result['output_files'])
        total_stats["pages_skipped"] += result['total_skipped']

    print(f"\n{'='*60}")
    print(f"总计: 生成 {total_stats['files_created']} 个文件, 跳过 {total_stats['pages_skipped']} 页")


if __name__ == "__main__":
    main()
