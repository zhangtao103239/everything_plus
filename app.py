import os
import tkinter as tk
from tkinter import ttk, messagebox
import tkinter.scrolledtext as scrolledtext
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import asyncio
import aiohttp
import chromadb
import json
from docx import Document
from PyPDF2 import PdfReader
import textract
from typing import Optional
import logging

CONFIG_PATH = ".config.json"
SUPPORTED_EXTENSIONS = {".docx", ".pdf"}  # 白名单


class Config:
    def __init__(self):
        if os.path.isfile(CONFIG_PATH):
            with open(CONFIG_PATH, encoding="utf8") as f:
                configMap = json.load(f)
                self.embedding_url = configMap["embedding_url"]
                self.embedding_api_key = configMap["embedding_api_key"]
                self.reranker_url = configMap["reranker_url"]
                self.reranker_api_key = configMap["reranker_api_key"]
                self.llm_api_url = configMap["llm_api_url"]
        else:
            self.embedding_url = ""
            self.embedding_api_key = ""
            self.reranker_url = ""
            self.reranker_api_key = ""
            self.llm_api_url = ""
            self.llm_api_key = ""


class FileProcessor:
    PROCESSED_RECORD = ".processed_files.json"

    def __init__(self, config: Config, logger):
        self.config = config
        # 创建Chroma客户端
        self.client = chromadb.PersistentClient(path=".chroma")  # 直接指定存储路径
        # 创建/获取集合
        self.collection = self.client.get_or_create_collection("file_chunks")
        self.processed_files = set()
        self.logger = logger
        self.load_processed_files()

    def load_processed_files(self):
        if os.path.exists(self.PROCESSED_RECORD):
            with open(self.PROCESSED_RECORD, "r") as f:
                self.processed_files = set(json.load(f))
        else:
            self.processed_files = set()

    def save_processed_files(self):
        with open(self.PROCESSED_RECORD, "w") as f:
            json.dump(list(self.processed_files), f)

    def scan_all_files(self, directory):
        for root_dir, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root_dir, file)
                if file_path not in self.processed_files:
                    self.process_file(file_path)
        self.save_processed_files()

    def is_excluded(self, path: str) -> bool:
        # 排除隐藏文件
        basePath = os.path.basename(path)
        if basePath.startswith(".") or basePath.startswith("~$"):
            return True

        # 检查扩展名是否在白名单
        _, ext = os.path.splitext(path)
        if ext.lower() not in SUPPORTED_EXTENSIONS:
            return True

        # 排除特定目录
        excluded_dirs = {"__pycache__", "node_modules", "venv", "Program Files"}
        for dir in excluded_dirs:
            if dir in path:
                return True
        return False

    async def embed_text(self, texts: list, batch_size=15) -> list:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            data = {"input": batch}
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.config.embedding_api_key}",
                    "Content-Type": "application/json",
                }
                async with session.post(
                    self.config.embedding_url, headers=headers, json=data
                ) as resp:
                    if resp.status == 413:
                        error_msg = await resp.text()
                        self.logger.error(f"413 Error: {error_msg}")
                        raise ValueError(f"413 Error: {error_msg}")
                    elif not resp.ok:
                        self.logger.error(
                            f"HTTP Error {resp.status}: {await resp.text()}"
                        )
                        raise RuntimeError(
                            f"HTTP Error {resp.status}: {await resp.text()}"
                        )
                    response_data = await resp.json()
                    all_embeddings.extend(response_data["data"])
                    self.logger.debug(
                        f"Embedded batch {i//batch_size+1}/{len(texts)//batch_size+1}"
                    )
        return all_embeddings

    def extract_text(self, file_path: str) -> Optional[str]:
        try:
            if file_path.endswith(".pdf"):
                reader = PdfReader(file_path)
                text = "\n".join(page.extract_text() for page in reader.pages)
            elif file_path.endswith(".docx"):
                doc = Document(file_path)
                text = "\n".join(para.text for para in doc.paragraphs)
            else:
                text = textract.process(file_path).decode("utf-8", errors="ignore")
            self.logger.info(f"Extracted text from {file_path}")
            return text
        except Exception as e:
            self.logger.error(f"Failed to extract text from {file_path}: {e}")
            return None

    def split_text(self, text: str) -> list:
        chunks = []
        chunk_size = 4096
        overlap = 1024
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i : i + chunk_size])
        self.logger.debug(f"Split into {len(chunks)} chunks")
        return chunks

    def process_file(self, file_path: str):
        if self.is_excluded(file_path) or file_path in self.processed_files:
            return
        self.logger.info(f"Processing file: {file_path}")
        text = self.extract_text(file_path)
        if not text:
            return
        chunks = self.split_text(text)
        try:
            embeddings = asyncio.run(self.embed_text(chunks))
        except Exception as e:
            self.logger.error(f"Embedding failed for {file_path}: {e}")
            return

        # 准备数据
        ids = [f"{file_path}_{i}" for i in range(len(chunks))]
        metadatas = [
            {"file_path": file_path, "chunk_index": i, "content": chunk[:500]}
            for i, chunk in enumerate(chunks)
        ]
        vectors = [emb["embedding"] for emb in embeddings]

        # 添加到Chroma
        try:
            self.collection.add(ids=ids, embeddings=vectors, metadatas=metadatas)
            self.processed_files.add(file_path)
            self.logger.info(f"Successfully processed {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to add to Chroma: {e}")


class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, processor: FileProcessor):
        self.processor = processor

    def on_modified(self, event):
        if not event.is_directory:
            self.processor.process_file(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            self.processor.process_file(event.src_path)


class SearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("智能内容搜索引擎")
        self.config = Config()
        self.processor = None
        self.observer = None
        self.logger = logging.getLogger("SearchApp")
        self.logger.setLevel(logging.DEBUG)
        self.create_widgets()
        if self.config.embedding_url != "":
            self.processor = FileProcessor(self.config, self.logger)
            messagebox.showinfo("成功", "配置已读取")
            self.logger.info("Configuration saved successfully")
            self.start_observer()

    def create_widgets(self):
        # 配置面板
        config_frame = ttk.LabelFrame(self.root, text="服务配置", padding=10)
        config_frame.pack(fill=tk.X, padx=10, pady=10)

        # Embedding服务
        ttk.Label(config_frame, text="Embedding URL:").grid(
            row=0, column=0, padx=5, pady=5
        )
        self.embedding_url = ttk.Entry(config_frame, width=40)
        self.embedding_url.grid(row=0, column=1, padx=5, pady=5)
        # 显示配置值
        self.embedding_url.insert(0, self.config.embedding_url)  # 初始化显示

        ttk.Label(config_frame, text="API Key:").grid(row=0, column=2, padx=5, pady=5)
        self.embedding_api_key = ttk.Entry(config_frame, width=30, show="*")
        self.embedding_api_key.grid(row=0, column=3, padx=5, pady=5)
        self.embedding_api_key.insert(0, self.config.embedding_api_key)  # 初始化显示

        # Reranker服务
        ttk.Label(config_frame, text="Reranker URL:").grid(
            row=1, column=0, padx=5, pady=5
        )
        self.reranker_url = ttk.Entry(config_frame, width=40)
        self.reranker_url.grid(row=1, column=1, padx=5, pady=5)
        self.reranker_url.insert(0, self.config.reranker_url)  # 初始化显示

        ttk.Label(config_frame, text="API Key:").grid(row=1, column=2, padx=5, pady=5)
        self.reranker_api_key = ttk.Entry(config_frame, width=30, show="*")
        self.reranker_api_key.grid(row=1, column=3, padx=5, pady=5)
        self.reranker_api_key.insert(0, self.config.reranker_api_key)  # 初始化显示

        # 保存配置按钮
        save_btn = ttk.Button(config_frame, text="保存配置", command=self.save_config)
        save_btn.grid(row=4, columnspan=4, pady=10)

        # 主操作面板
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 搜索区域
        search_frame = ttk.LabelFrame(main_frame, text="搜索", padding=10)
        search_frame.pack(fill=tk.X)

        self.search_var = tk.StringVar()
        ttk.Entry(search_frame, textvariable=self.search_var, width=50).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(search_frame, text="搜索", command=self.perform_search).pack(
            side=tk.LEFT, padx=5
        )

        # 结果区域
        self.results = ttk.Treeview(main_frame, columns=('文件', '内容'), 
                          show='headings', height=15)
        self.results.column('内容', width=400)  # 扩展列宽
        self.results.heading('文件', text='文件路径')
        self.results.heading('内容', text='内容预览')
        # 添加详细预览框
        self.preview = scrolledtext.ScrolledText(main_frame, height=10)
        self.preview.pack(fill=tk.BOTH, expand=True, pady=5)
        # 状态栏
        self.status = ttk.Label(self.root, text="就绪", anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
        # 新增日志显示区域
        log_frame = ttk.LabelFrame(self.root, text="系统日志", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.log_display = scrolledtext.ScrolledText(
            log_frame, state="disabled", height=10
        )
        self.log_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # 绑定选择事件
        def on_select(event):
            selected = self.results.selection()
            if selected:
                item = self.results.item(selected[0])
                self.preview.delete(1.0, tk.END)
                self.preview.insert(tk.END, item['values'][1])
        self.results.bind('<<TreeviewSelect>>', on_select)
        class GuiLogHandler(logging.Handler):
            def __init__(self, text_widget):
                super().__init__()
                self.text_widget = text_widget

            def emit(self, record):
                msg = self.format(record)
                self.text_widget.configure(state="normal")
                self.text_widget.insert(tk.END, msg + "\n")
                self.text_widget.configure(state="disabled")
                self.text_widget.yview(tk.END)  # 自动滚动到底部

        gui_handler = GuiLogHandler(self.log_display)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        gui_handler.setFormatter(formatter)
        self.logger.addHandler(gui_handler)

    def save_config(self):
        self.config.embedding_url = self.embedding_url.get()
        self.config.embedding_api_key = self.embedding_api_key.get()
        self.config.reranker_url = self.reranker_url.get()
        self.config.reranker_api_key = self.reranker_api_key.get()

        # 将配置保存到字典中
        config_dict = {
            "embedding_url": self.config.embedding_url,
            "embedding_api_key": self.config.embedding_api_key,
            "reranker_url": self.config.reranker_url,
            "reranker_api_key": self.config.reranker_api_key,
            "llm_api_url": self.config.llm_api_url,
        }
        # 将配置字典写入 JSON 文件
        with open(CONFIG_PATH, "w", encoding="utf8") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=4)
        self.processor = FileProcessor(self.config, self.logger)
        messagebox.showinfo("成功", "配置已保存")
        self.logger.info("Configuration saved successfully")
        self.start_observer()

    def start_observer(self):
        if self.observer:
            self.observer.stop()
        self.observer = Observer()
        self.observer.schedule(
            FileChangeHandler(self.processor), path="/", recursive=True
        )
        self.observer.start()
        self.logger.info("File monitoring started")
        # self.processor.scan_all_files("/")  # 初始扫描

    def perform_search(self):
        """执行搜索操作"""
        query = self.search_var.get()
        if not query:
            return

        # 生成查询向量
        async def get_query_vector():
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {self.config.embedding_api_key}"}
                data = {"input": [query]}
                async with session.post(
                    self.config.embedding_url, headers=headers, json=data
                ) as resp:
                    return await resp.json()

        embedding = asyncio.run(get_query_vector())
        query_vector = embedding["data"][0]["embedding"]

        # 执行搜索
        results = self.processor.collection.query(
            query_embeddings=[query_vector], n_results=10
        )

        # 显示结果
        self.results.delete(*self.results.get_children())
        for i in range(len(results["ids"][0])):
            self.results.insert(
                "",
                "end",
                values=(
                    results["metadatas"][0][i]["file_path"],
                    results["metadatas"][0][i]["content"].replace("\n", "  "),
                ),
            )
        self.status.config(text=f"找到 {len(results['ids'][0])} 个结果")
        self.logger.info(
            f"Searching for '{query}' returned {len(results['ids'][0])} results"
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = SearchApp(root)
    root.mainloop()
