from dateutil import parser as dparser
import re
import pandas as pd
import emoji
from urlextract import URLExtract

extractor = URLExtract()

PHONE_RE = re.compile(r'(\+?\d[\d\-\s]{5,}\d)')
EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+')
URL_RE = re.compile(r'https?://\S+|www\.\S+')

TIMESTAMP_USER_RE = re.compile(r"""
    ^\s*
    (?P<ts>
      [0-9]{1,2}[/\-.][0-9]{1,2}[/\-.][0-9]{2,4}
      [,\sT\-:APMapm0-9/\.]*
    )
    \s*-\s*
    (?P<rest>.*)
    """, re.VERBOSE)

def redact_pii(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = PHONE_RE.sub('[PHONE]', text)
    text = EMAIL_RE.sub('[EMAIL]', text)
    text = URL_RE.sub('[URL]', text)
    return text

def try_parse_line(line: str):
    m = TIMESTAMP_USER_RE.match(line)
    if not m:
        return None
    ts_raw = m.group('ts').strip()
    rest = m.group('rest').strip()
    ts = None
    for dayfirst in (False, True):
        try:
            ts = dparser.parse(ts_raw, fuzzy=True, dayfirst=dayfirst)
            break
        except Exception:
            ts = None
    if ts is None:
        return None
    if ':' in rest:
        author, msg = rest.split(':', 1)
        return ts, author.strip(), msg.strip()
    else:
        return ts, None, rest

def detect_media_type(message: str):
    msg_lower = message.lower().strip()
    if any(ph in msg_lower for ph in ["<media omitted>", "<image omitted>", "image omitted", "video omitted", "audio omitted", "file omitted"]):
        if "image" in msg_lower:
            return "image"
        if "video" in msg_lower:
            return "video"
        if "audio" in msg_lower:
            return "audio"
        return "media"
    if "<attached" in msg_lower or "attachment" in msg_lower:
        return "attachment"
    return None

def count_emojis(text: str) -> int:
    if not isinstance(text, str):
        return 0
    try:
        return sum(1 for ch in text if ch in emoji.UNICODE_EMOJI_ENGLISH)
    except Exception:
        return sum(1 for ch in text if ch in emoji.UNICODE_EMOJI)

def count_links(text: str) -> int:
    if not isinstance(text, str):
        return 0
    try:
        return len(extractor.find_urls(text))
    except Exception:
        return len(re.findall(r'https?://\S+|www\.\S+', text))

def parse_chat_from_lines(lines):
    data = []
    current = {"datetime": None, "user": None, "message": "", "is_system_message": False}
    for raw_line in lines:
        line = raw_line.rstrip('\n')
        parsed = try_parse_line(line)
        if parsed:
            if current["datetime"] is not None:
                data.append(current.copy())
            ts, author, msg = parsed
            current = {
                "datetime": ts,
                "user": author,
                "message": msg,
                "is_system_message": (author is None)
            }
        else:
            if current["message"] is None:
                current["message"] = line
            else:
                current["message"] = current["message"] + "\n" + line
    if current["datetime"] is not None:
        data.append(current.copy())
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df['message'] = df['message'].astype(str).apply(lambda x: redact_pii(x))
    df['media_type'] = df['message'].apply(detect_media_type)
    df['message_length'] = df['message'].apply(lambda x: len(x))
    df['emoji_count'] = df['message'].apply(count_emojis)
    df['link_count'] = df['message'].apply(count_links)
    df['date'] = df['datetime'].dt.date
    df['time'] = df['datetime'].dt.time
    df['hour'] = df['datetime'].dt.hour
    df['weekday'] = df['datetime'].dt.day_name()
    df = df.sort_values('datetime').reset_index(drop=True)
    df['prev_user'] = df['user'].shift(1)
    df['prev_dt'] = df['datetime'].shift(1)
    def compute_response(row):
        if pd.isna(row['prev_dt']) or row['user'] == row['prev_user'] or pd.isna(row['prev_user']):
            return None
        diff = row['datetime'] - row['prev_dt']
        return int(diff.total_seconds())
    df['response_time_seconds'] = df.apply(compute_response, axis=1)
    df = df.drop(columns=['prev_user', 'prev_dt'])
    return df

def parse_chat_from_file(filepath_or_fileobj):
    if hasattr(filepath_or_fileobj, 'read'):
        content = filepath_or_fileobj.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8', errors='replace')
        lines = content.splitlines()
    else:
        with open(filepath_or_fileobj, 'r', encoding='utf-8', errors='replace') as fh:
            lines = fh.readlines()
    return parse_chat_from_lines(lines)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        df = parse_chat_from_file(sys.argv[1])
        print(df.head().to_dict(orient='records'))
    else:
        print("Usage: python preprocessor.py <chat.txt>")
