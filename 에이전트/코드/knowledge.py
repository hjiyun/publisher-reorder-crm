"""도서 지식 에이전트: 제목으로 도서 종류를 분류하고(A·B 독립 분류 → 엇갈린 책만 판정), 자격시험 시행 월을 웹에서 찾는다.

에이전트에게는 도서 제목·분야·출간 연월만 보여 준다(2026-10-06 사용자 허락). ISBN 대신 내부 번호(B001…)를 쓴다.
결과는 도서별 종류표와 시험 일정표이고, improve.py의 변수 묶음(book_type, type_season, exam_calendar)에 쓰인다.
"""
import json

import pandas as pd

import agents as ag

EXAM_TYPES = ['DIAT', 'ITQ', 'GTQ', '컴퓨터활용능력', '워드프로세서', '정보처리', '기타 자격시험', '해당 없음']
SUBJECTS = ['오피스(엑셀·한글·워드·파워포인트)', '멀티미디어·그래픽', '코딩·프로그래밍', 'AI·데이터', '디자인 도구(캔바 등)', '기타']
AUDIENCE = ['초등', '중등', '고등', '성인·일반', '혼합·불명']
S = ag._strict
CHUNK = 60  # 분류가 한 번에 받는 도서 수
BOOK = S({'ref': {'type': 'string'}, 'exam_type': {'type': 'string', 'enum': EXAM_TYPES}, 'subject': {'type': 'string', 'enum': SUBJECTS},
          'audience': {'type': 'string', 'enum': AUDIENCE}, 'series': {'type': 'string'}, 'confidence': {'type': 'string', 'enum': ['high', 'medium', 'low']}})
CLASSIFY_OUT = S({'books': {'type': 'array', 'items': BOOK}})
JUDGE_OUT = S({'books': {'type': 'array', 'items': S({**{k: v for k, v in BOOK['properties'].items() if k != 'confidence'}, 'reason': {'type': 'string'}})}})
CAL_OUT = S({'exams': {'type': 'array', 'items': S({'exam_type': {'type': 'string', 'enum': EXAM_TYPES[:-1]}, 'months': {'type': 'array', 'items': {'type': 'integer'}},
                                                     'basis': {'type': 'string'}, 'sources': {'type': 'array', 'items': {'type': 'string'}}})},
             'school_note': {'type': 'string'}})

BASE = '''당신은 한국 교육·IT 도서 전문가입니다. 한 출판사의 도서 목록(내부 번호, 제목, 서점 분야, 출간 연월)을 보고 도서마다 아래를 정하세요.
- exam_type: 이 책이 대비하는 자격시험. 시험 대비서가 아니면 '해당 없음'. DIAT·ITQ·GTQ·컴퓨터활용능력·워드프로세서·정보처리 밖의 자격시험은 '기타 자격시험'.
- subject: 주제. audience: 주 독자 학년. 제목에 근거가 없으면 '혼합·불명'.
- series: 제목에서 드러나는 시리즈명(예: 발자취, 우당탕탕). 없으면 빈 문자열.
- confidence: 제목만으로 얼마나 확실한지.
모든 도서를 빠짐없이 한 번씩, 받은 내부 번호(ref) 그대로 답하고, 최종 답은 지정된 JSON 형식으로만 내세요.'''
ROLES = {
    'classifier_A': BASE + '\n당신은 분류가 A입니다. 제목의 시험명·프로그램명·버전 같은 직접적인 단서를 우선합니다.',
    'classifier_B': BASE + '\n당신은 분류가 B입니다. 같은 시리즈 도서들이 어떻게 구성되는지(시리즈 전체의 성격)를 함께 고려합니다.',
    'classifier_judge': BASE.replace('도서마다 아래를 정하세요', '두 분류가의 판단이 엇갈린 도서만 보고 최종 분류를 정하세요') +
                        '\n각 도서에 두 분류가의 답이 함께 주어집니다. reason에 판단 근거를 짧게 적으세요.',
    'exam_calendar': '''당신은 한국 IT 자격시험 일정 조사원입니다. 웹 검색으로 DIAT, ITQ, GTQ, 컴퓨터활용능력, 워드프로세서, 정보처리 시험의 정기 시행 월(연중 몇 월에 시험이 있는지)을 찾으세요.
- 최근 2~3년의 공식 일정(시행 기관 공지)을 근거로 하고, 해마다 반복되는 정기 시행 월만 months(1~12)에 적으세요. 상시 시험처럼 거의 매달 있으면 그 달들을 모두 적고 basis에 그렇게 쓰세요.
- sources에 실제로 확인한 페이지 주소를 적으세요. 확인하지 못한 시험은 months를 비우고 basis에 '확인 못함'이라고 쓰세요. 추측으로 채우지 마세요.
- school_note에 초·중·고 학기 시작 월과 방과후 수업 시작 시기를 한두 문장으로 적으세요.
최종 답은 지정된 JSON 형식으로만 내세요.''',
}
for role, prompt in ROLES.items():
    ag.ROLES[role] = prompt
    ag.OUT[role] = {'classifier_judge': JUDGE_OUT, 'exam_calendar': CAL_OUT}.get(role, CLASSIFY_OUT)
    ag.ROLE_TOOLS[role] = []
    ag.PREFIX[role] = {'classifier_A': 'KA', 'classifier_B': 'KB', 'classifier_judge': 'KJ', 'exam_calendar': 'KC'}[role]
ag.ROLE_BUILTINS['exam_calendar'] = ['WebSearch']


def catalog(books):
    b = books[books.isbn.astype(str).str.len() > 0].drop_duplicates('isbn').sort_values('isbn').reset_index(drop=True)
    b['ref'] = [f'B{i + 1:03d}' for i in range(len(b))]
    b['pub_month'] = pd.to_datetime(b.pub).dt.strftime('%Y-%m').fillna('')
    return b[['ref', 'isbn', 'title', 'genre', 'pub_month']]


def classify(agents, books, say=print):
    """(도서별 종류표 DataFrame[isbn, ref, title, exam_type, subject, audience, series, agreed, source], 요약 dict)."""
    cat = catalog(books)
    line = lambda r: f'{r.ref} | {r.title} | {r.genre} | {r.pub_month}'
    fp = ag.hashlib.sha256('\n'.join(line(r) for r in cat.itertuples()).encode()).hexdigest()[:16]
    outs, metas = {}, []
    for who in ('classifier_A', 'classifier_B'):  # 한 번에 CHUNK종씩 (출력 길이 한도 대비)
        outs[who], cost = {}, 0.0
        for i in range(0, len(cat), CHUNK):
            part = cat.iloc[i:i + CHUNK]
            user = (f'도서 {len(part)}종입니다(전체 {len(cat)}종 중 {i + 1}~{i + len(part)}번째). 형식: 내부 번호 | 제목 | 서점 분야 | 출간 연월\n'
                    + '\n'.join(line(r) for r in part.itertuples()))
            out, _, meta = agents.run(who, None, user, fp)
            outs[who].update({b['ref']: b for b in out['books'] if b['ref'] in set(part.ref)})
            metas.append(meta)
            cost += 0 if meta.get('cached') else meta['cost_usd']
        say(f'지식: {who} 완료 · {len(outs[who])}/{len(cat)}종 · ${cost:.2f}')
    keys = ('exam_type', 'subject', 'audience')
    rows, disputed = {}, []
    for ref in cat.ref:
        a, b = outs['classifier_A'].get(ref), outs['classifier_B'].get(ref)
        if a and b and all(a[k] == b[k] for k in keys):
            rows[ref] = dict(a, series=a['series'] or b['series'], agreed=1, source='A·B 일치')
        else:
            disputed.append((ref, a, b))
    if disputed:
        title = cat.set_index('ref').title
        text = '\n'.join(f'{ref} | {title[ref]} | A: {json.dumps({k: a.get(k) for k in keys + ("series",)}, ensure_ascii=False) if a else "답 없음"} | '
                         f'B: {json.dumps({k: b.get(k) for k in keys + ("series",)}, ensure_ascii=False) if b else "답 없음"}' for ref, a, b in disputed)
        out, _, meta = agents.run('classifier_judge', None, f'판단이 엇갈린 도서 {len(disputed)}종입니다.\n{text}', fp)
        metas.append(meta)
        say(f'지식: classifier_judge 완료 · {len(disputed)}종 판정 · {meta["seconds"]}s · ${meta["cost_usd"]:.2f}')
        for b in out['books']:
            if b['ref'] in title.index:
                rows[b['ref']] = dict(b, agreed=0, source=f'판정: {b["reason"]}')
    for ref, a, b in disputed:  # 판정에서도 빠진 도서는 A의 답, 그것도 없으면 '해당 없음'
        rows.setdefault(ref, dict(a or b or dict(exam_type='해당 없음', subject='기타', audience='혼합·불명', series=''), agreed=0, source='판정 누락'))
    k = cat.merge(pd.DataFrame([dict(ref=r, **{c: v[c] for c in ('exam_type', 'subject', 'audience', 'series', 'agreed', 'source')}) for r, v in rows.items()]), on='ref')
    summary = dict(books=len(cat), agreed=int(k.agreed.sum()), agreement_rate=float(k.agreed.mean()), disputed=len(disputed),
                   cost_usd=sum(m['cost_usd'] for m in metas if not m.get('cached')))
    return k, summary


def exam_calendar(agents, say=print):
    out, _, meta = agents.run('exam_calendar', None, '자격시험 정기 시행 월을 조사해 주세요.', 'calendar-v1')
    say(f'지식: exam_calendar 완료 · {meta["seconds"]}s · ${meta["cost_usd"]:.2f}{" (저장분)" if meta.get("cached") else ""}')
    cal = {e['exam_type']: sorted({m for m in e['months'] if 1 <= m <= 12}) for e in out['exams']}
    return cal, out, meta
