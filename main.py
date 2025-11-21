import os
import re
import json
from typing import List, Dict, Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import tldextract

from database import create_document
from schemas import BrandAnalysis, Review, SentimentScores, SEOBasics

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    url: HttpUrl
    limit_reviews: int = 25


def fetch_url(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; FlamesBlueBot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=12)
        if resp.status_code == 200 and 'text/html' in resp.headers.get('Content-Type', ''):
            return resp.text
        return ""
    except Exception:
        return ""


def extract_domain(url: str) -> str:
    ext = tldextract.extract(url)
    domain = ".".join([p for p in [ext.domain, ext.suffix] if p])
    return domain


def parse_basic_seo(html: str) -> SEOBasics:
    soup = BeautifulSoup(html, 'html.parser')
    title = (soup.title.string.strip() if soup.title and soup.title.string else None)
    desc_tag = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
    description = desc_tag.get('content').strip() if desc_tag and desc_tag.get('content') else None

    keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
    keywords = []
    if keywords_tag and keywords_tag.get('content'):
        keywords = [k.strip() for k in keywords_tag['content'].split(',') if k.strip()]

    headings = { 'h1': [], 'h2': [], 'h3': [] }
    for level in headings.keys():
        for h in soup.find_all(level):
            text = h.get_text(" ", strip=True)
            if text:
                headings[level].append(text)

    return SEOBasics(title=title, description=description, keywords=keywords, h1=headings['h1'], h2=headings['h2'], h3=headings['h3'])


def extract_social_links(html: str) -> Dict[str, str]:
    soup = BeautifulSoup(html, 'html.parser')
    links = {}
    for a in soup.find_all('a', href=True):
        href = a['href']
        for name, pattern in {
            'twitter': r'twitter.com|x.com',
            'facebook': r'facebook.com',
            'instagram': r'instagram.com',
            'linkedin': r'linkedin.com',
            'youtube': r'youtube.com|youtu.be',
            'tiktok': r'tiktok.com'
        }.items():
            if re.search(pattern, href, re.I):
                links[name] = href
    return links


def extract_colors(html: str) -> List[str]:
    # naive color extraction from inline styles and style tags
    colors = set()
    try:
        soup = BeautifulSoup(html, 'html.parser')
        styles = " ".join([s.get_text(" ") for s in soup.find_all('style')])
        inline_styles = " ".join([tag.get('style', '') for tag in soup.find_all(True) if tag.get('style')])
        blob = styles + " " + inline_styles
        for m in re.findall(r"#(?:[0-9a-fA-F]{3}){1,2}", blob):
            colors.add(m.lower())
        for m in re.findall(r"rgb\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)", blob):
            colors.add(m)
    except Exception:
        pass
    return list(colors)[:8]


def guess_favicon(html: str, base_url: str) -> str | None:
    try:
        soup = BeautifulSoup(html, 'html.parser')
        icon = soup.find('link', rel=lambda x: x and 'icon' in x.lower())
        if icon and icon.get('href'):
            href = icon['href']
            if href.startswith('http'):  # absolute
                return href
            # relative
            pr = urlparse(base_url)
            base = f"{pr.scheme}://{pr.netloc}"
            if href.startswith('/'):
                return base + href
            else:
                return base + '/' + href
    except Exception:
        pass
    return None


SEARCH_ENDPOINT = "https://www.google.com/search"


def fetch_search_snippets(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    # Basic search fallback using startpage-like approach is not available; we'll attempt Google HTML
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        }
        params = {"q": query, "num": min(limit, 20)}
        r = requests.get(SEARCH_ENDPOINT, params=params, headers=headers, timeout=12)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, 'html.parser')
        results = []
        for g in soup.select('div.g'):
            a = g.find('a', href=True)
            if not a:
                continue
            title = g.find('h3')
            snippet = g.find('span', class_='aCOpRe') or g.find('div', class_='VwiC3b')
            results.append({
                'title': title.get_text(strip=True) if title else a.get_text(strip=True),
                'url': a['href'],
                'snippet': snippet.get_text(" ", strip=True) if snippet else ''
            })
            if len(results) >= limit:
                break
        return results
    except Exception:
        return []


def collect_reviews(domain: str, brand: str, max_items: int = 25) -> List[Review]:
    reviews: List[Review] = []
    queries = [
        f"site:trustpilot.com {brand}",
        f"site:g2.com {brand}",
        f"site:reddit.com {brand} reviews",
        f"site:news.ycombinator.com {brand}",
        f"{brand} reviews",
        f"{domain} reviews"
    ]
    for q in queries:
        for r in fetch_search_snippets(q, limit=5):
            # Try to fetch page and extract simple review-like texts
            html = fetch_url(r['url'])
            if not html:
                continue
            soup = BeautifulSoup(html, 'html.parser')
            # naive extraction: paragraph texts that contain brand name
            paras = [p.get_text(" ", strip=True) for p in soup.find_all('p')]
            for p in paras:
                if len(p) > 60 and (brand.lower() in p.lower() or 'review' in p.lower()):
                    reviews.append(Review(source=r['url'], text=p))
                    if len(reviews) >= max_items:
                        return reviews
    return reviews


def simple_sentiment(texts: List[str]) -> SentimentScores:
    # Very naive rule-based sentiment due to no external NLP packages added
    pos_words = set(["good","great","excellent","love","amazing","awesome","fast","easy","helpful","recommend","best","perfect","happy","satisfied"]) 
    neg_words = set(["bad","terrible","awful","hate","slow","difficult","buggy","problem","worst","poor","disappointed","angry","frustrated"]) 
    pos = neg = neu = 0
    for t in texts:
        tl = t.lower()
        pw = sum(1 for w in pos_words if w in tl)
        nw = sum(1 for w in neg_words if w in tl)
        if pw > nw:
            pos += 1
        elif nw > pw:
            neg += 1
        else:
            neu += 1
    total = max(1, pos + neg + neu)
    overall = (pos - neg) / total
    return SentimentScores(positive=pos/total, negative=neg/total, neutral=neu/total, overall=overall)


def extract_keywords(text: str, top_k: int = 15) -> List[str]:
    # naive keyword extraction: top frequent non-stopwords
    stop = set("""a an the and or but if while with without of in on for to from at by as is are was were be been being this that these those it its it's they them you your we us our i me my he she him her his hers their theirs not no yes do does did can will would should could just more most less least very really over under into out about around within across use using used website brand product service company customers users online review reviews""".split())
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    freq: Dict[str,int] = {}
    for w in words:
        if w in stop:
            continue
        freq[w] = freq.get(w,0)+1
    return [w for w,_ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_k]]


@app.post("/api/analyze", response_model=BrandAnalysis)
def analyze(req: AnalyzeRequest):
    url = str(req.url)
    html = fetch_url(url)
    if not html:
        raise HTTPException(status_code=400, detail="Unable to fetch the website content.")

    domain = extract_domain(url)
    seo = parse_basic_seo(html)
    social = extract_social_links(html)
    colors = extract_colors(html)
    favicon = guess_favicon(html, url)

    all_text = BeautifulSoup(html, 'html.parser').get_text(" ", strip=True)
    kw = extract_keywords(all_text)

    brand = domain.split('.')[0]
    reviews = collect_reviews(domain, brand, max_items=req.limit_reviews)
    review_texts = [r.text for r in reviews if r.text]
    sentiment = simple_sentiment(review_texts)

    analysis = BrandAnalysis(
        url=url,
        domain=domain,
        favicon=favicon,
        tech=[],
        social_links=social,
        color_palette=colors,
        seo=seo,
        keywords=kw,
        reviews=reviews,
        sentiment=sentiment,
        summary=(
            f"We analyzed {domain}. The page title is '{seo.title or 'N/A'}'. "
            f"Found {len(seo.h1)} H1s and {len(reviews)} external review snippets. "
            f"Overall sentiment is {round((sentiment.overall+1)/2*100)}%."
        ),
        raw_samples={
            'search_queries_used': [
                f"site:trustpilot.com {brand}",
                f"site:g2.com {brand}",
                f"site:reddit.com {brand} reviews",
                f"site:news.ycombinator.com {brand}",
                f"{brand} reviews",
                f"{domain} reviews"
            ]
        }
    )

    try:
        create_document('brandanalysis', analysis)
    except Exception:
        # database optional; ignore errors if not configured
        pass

    return analysis


@app.get("/")
def read_root():
    return {"message": "Brand Analyzer API running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        from database import db
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except ImportError:
        response["database"] = "❌ Database module not found"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
