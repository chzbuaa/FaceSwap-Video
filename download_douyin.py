#!/usr/bin/env python3
"""
抖音视频下载脚本

支持：
  - v.douyin.com 短链、www.douyin.com / www.iesdouyin.com 分享页
  - 从 App 复制的整段分享文案中自动提取链接
  - 默认解析分享页内嵌数据直链下载（无需登录）
  - 可选 yt-dlp + 浏览器 Cookie（页面解析失败或需更高画质时使用）

依赖：
  Python 3.9+（标准库即可）
  可选：pip install yt-dlp

示例：
  python download_douyin.py
  python download_douyin.py "https://v.douyin.com/C_JND893-nI/"
  python download_douyin.py --ytdlp --cookies-from-browser chrome
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parent

# 默认：用户提供的分享文案（脚本会自动提取其中的链接）
DEFAULT_SHARE = (
    'https://v.douyin.com/OW_r6aB4VBo/ b@N.JI 01/01 EHi:/ :8pm'
)
OUTPUT_DIR = PROJECT_ROOT / "downloads" / "douyin"

MOBILE_UA = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1"
)

DOUYIN_URL_RE = re.compile(
    r"https?://(?:v\.douyin\.com|www\.douyin\.com|www\.iesdouyin\.com)/[^\s\u4e00-\u9fff]+",
    re.I,
)
ROUTER_DATA_RE = re.compile(
    r"window\._ROUTER_DATA\s*=\s*(\{.*?\})\s*;?\s*</script>",
    re.S,
)


@dataclass
class DouyinVideo:
    aweme_id: str
    title: str
    play_url: str
    page_url: str


def extract_share_url(text: str) -> str:
    """从分享文案或纯链接中提取抖音 URL。"""
    text = text.strip()
    m = DOUYIN_URL_RE.search(text)
    if m:
        url = m.group(0).rstrip(".,;)")
        if not url.endswith("/"):
            url += "/"
        return url
    if text.startswith("http"):
        return text
    raise ValueError(f"未在文本中找到抖音链接：{text[:120]}...")


def sanitize_filename(name: str, max_len: int = 80) -> str:
    name = re.sub(r'[<>:"/\\|?*\n\r\t]', "_", name).strip(" .")
    if not name:
        name = "douyin_video"
    return name[:max_len]


def _request(url: str, *, referer: str | None = None) -> tuple[str, bytes]:
    headers = {"User-Agent": MOBILE_UA}
    if referer:
        headers["Referer"] = referer
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read()
        return resp.geturl(), body


def fetch_share_page(url: str) -> tuple[str, str]:
    """请求分享页，返回 (最终 URL, HTML)。"""
    final, raw = _request(url)
    html = raw.decode("utf-8", errors="replace")
    host = urlparse(final).netloc.lower()
    if "douyin.com" not in host:
        raise RuntimeError(f"短链未跳转到抖音页面，最终 URL：{final}")
    return final, html


def _item_from_router(data: dict) -> dict:
    loader = data.get("loaderData") or {}
    for val in loader.values():
        if not isinstance(val, dict):
            continue
        info = val.get("videoInfoRes")
        if not isinstance(info, dict):
            continue
        items = info.get("item_list") or []
        if items:
            return items[0]
    raise RuntimeError("分享页中未找到视频数据（页面结构可能已变更）")


def parse_video_from_html(page_url: str, html: str) -> DouyinVideo:
    m = ROUTER_DATA_RE.search(html)
    if not m:
        raise RuntimeError("分享页中未找到 _ROUTER_DATA，可尝试 --ytdlp")
    data = json.loads(m.group(1))
    item = _item_from_router(data)
    video = item.get("video") or {}
    play_addr = video.get("play_addr") or {}
    url_list = play_addr.get("url_list") or []
    if not url_list:
        raise RuntimeError("未找到视频播放地址")
    play_url = url_list[0]
    aweme_id = str(item.get("aweme_id") or item.get("awemeId") or "")
    title = (
        item.get("desc")
        or (item.get("share_info") or {}).get("share_title")
        or f"douyin_{aweme_id or 'video'}"
    )
    return DouyinVideo(
        aweme_id=aweme_id,
        title=title.strip(),
        play_url=play_url,
        page_url=page_url,
    )


def to_no_watermark_play_url(play_url: str) -> str:
    """playwm 带水印，改为 play 可获取无水印直链（抖音 CDN 302）。"""
    return play_url.replace("/playwm/", "/play/").replace("playwm", "play", 1)


def resolve_direct_mp4_url(play_url: str, referer: str) -> str:
    """跟随 play/playwm 接口 302，得到 douyinvod 等 CDN 直链。"""
    play_url = to_no_watermark_play_url(play_url)
    headers = {"User-Agent": MOBILE_UA, "Referer": referer}
    req = urllib.request.Request(play_url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.geturl()


def download_file(
    url: str,
    dest: Path,
    *,
    referer: str,
    chunk_size: int = 256 * 1024,
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": MOBILE_UA, "Referer": referer}
    req = urllib.request.Request(url, headers=headers)
    total = 0
    with urllib.request.urlopen(req, timeout=120) as resp:
        length = resp.headers.get("Content-Length")
        total_expected = int(length) if length else None
        with dest.open("wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
                if total_expected and total % (1024 * 1024) < chunk_size:
                    pct = total * 100 // total_expected
                    print(f"\r  已下载 {total / 1024 / 1024:.1f} MB ({pct}%)", end="", flush=True)
                elif total % (5 * 1024 * 1024) < chunk_size:
                    print(f"\r  已下载 {total / 1024 / 1024:.1f} MB", end="", flush=True)
    print()
    return dest


def find_yt_dlp() -> str | None:
    for name in ("yt-dlp", "yt-dlp.exe"):
        p = shutil.which(name)
        if p:
            return p
    return None


def download_with_ytdlp(
    url: str,
    output_dir: Path,
    *,
    output_name: str | None = None,
    cookies_browser: str | None = None,
) -> Path:
    yt_dlp = find_yt_dlp()
    if not yt_dlp:
        raise RuntimeError(
            "未找到 yt-dlp，请安装：pip install yt-dlp\n"
            "或省略 --ytdlp，使用默认页面解析下载"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_name:
        out_tpl = str(output_dir / f"{sanitize_filename(output_name)}.%(ext)s")
    else:
        out_tpl = str(output_dir / "%(title)s.%(ext)s")

    cmd = [
        yt_dlp,
        "--no-playlist",
        "-o",
        out_tpl,
        "--retries",
        "5",
        "--fragment-retries",
        "5",
    ]
    if cookies_browser:
        cmd.extend(["--cookies-from-browser", cookies_browser])
    cmd.append(url)

    print("使用 yt-dlp 下载...")
    r = subprocess.run(cmd, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            "yt-dlp 下载失败。抖音常需 Cookie，可尝试：\n"
            "  --ytdlp --cookies-from-browser chrome"
        )

    files = sorted(output_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        files = sorted(output_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise RuntimeError(f"yt-dlp 完成但未在 {output_dir} 找到文件")
    return files[0]


def download_video(
    share_text: str,
    output_dir: Path,
    *,
    output_name: str | None = None,
    use_ytdlp: bool = False,
    cookies_browser: str | None = None,
) -> Path:
    url = extract_share_url(share_text)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"分享链接：{url}")

    if use_ytdlp:
        return download_with_ytdlp(
            url,
            output_dir,
            output_name=output_name,
            cookies_browser=cookies_browser,
        )

    page_url, html = fetch_share_page(url)
    print(f"页面地址：{page_url}")

    info = parse_video_from_html(page_url, html)
    print(f"标题：{info.title}")
    if info.aweme_id:
        print(f"作品 ID：{info.aweme_id}")

    direct = resolve_direct_mp4_url(info.play_url, page_url)
    print(f"直链：{direct[:90]}...")

    safe = sanitize_filename(output_name or info.title)
    if info.aweme_id and info.aweme_id not in safe:
        safe = f"{safe}_{info.aweme_id}"
    dest = output_dir / f"{safe}.mp4"

    if dest.is_file() and dest.stat().st_size > 0:
        print(f"已存在，跳过：{dest}")
        return dest

    print(f"保存到：{dest}")
    print("开始下载...")
    download_file(direct, dest, referer=page_url)
    return dest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="下载抖音视频（支持分享文案自动提取链接）")
    p.add_argument(
        "share",
        nargs="?",
        default=DEFAULT_SHARE,
        help="抖音分享链接或整段分享文案",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        default=str(OUTPUT_DIR),
        help=f"输出目录（默认：{OUTPUT_DIR}）",
    )
    p.add_argument(
        "-n",
        "--name",
        default="",
        help="指定输出文件名（不含扩展名）",
    )
    p.add_argument(
        "--ytdlp",
        action="store_true",
        help="改用 yt-dlp 下载（通常需浏览器 Cookie）",
    )
    p.add_argument(
        "--cookies-from-browser",
        metavar="BROWSER",
        default="",
        help="yt-dlp 从浏览器读取 Cookie，如 chrome / safari / edge",
    )
    p.add_argument(
        "--resolve-only",
        action="store_true",
        help="仅解析页面与直链，不下载",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    share = args.share.strip()

    try:
        url = extract_share_url(share)
        if args.resolve_only:
            page_url, html = fetch_share_page(url)
            info = parse_video_from_html(page_url, html)
            direct = resolve_direct_mp4_url(info.play_url, page_url)
            print(f"page: {page_url}")
            print(f"title: {info.title}")
            print(f"aweme_id: {info.aweme_id}")
            print(f"mp4: {direct}")
            return 0

        out = download_video(
            share,
            Path(args.output_dir),
            output_name=args.name.strip() or None,
            use_ytdlp=args.ytdlp,
            cookies_browser=args.cookies_from_browser.strip() or None,
        )
        size_mb = out.stat().st_size / 1024 / 1024
        print(f"\n下载完成：{out}")
        print(f"大小：{size_mb:.2f} MB")
        return 0
    except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError, ValueError) as e:
        print(f"错误：{e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
