import os
import json
import shutil
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from tqdm import tqdm
import tiktoken
import gradio as gr

# 환경 변수 및 설정
load_dotenv()
folder_path = "C:\\Users\\gram\\Desktop\\langchain_trytry\\sdf_sample"

# ─── 여기를 절대경로로 바꾸었습니다 ────────────────────────────────────
index_dir = "C:\\Users\\gram\\Desktop\\langchain_trytry\\faiss_index"
processed_files_path = os.path.join(index_dir, "processed_files.json")
faiss_index_file = os.path.join(index_dir, "index.faiss")
# ───────────────────────────────────────────────────────────────────────────

CODE_VERSION = 2

# TSV 파일 자동 병합
tsv_files = [f for f in os.listdir(folder_path) if f.endswith('.tsv')]
if len(tsv_files) < 2:
    raise FileNotFoundError(f"폴더 '{folder_path}'에 최소 2개의 TSV 파일이 필요합니다: {tsv_files}")
file1, file2 = tsv_files[:2]

# ─── 여기에 병합 로직을 추가했습니다 ─────────────────────────────────────────
df1 = pd.read_csv(os.path.join(folder_path, file1), sep="\t", encoding="utf-8")
df2 = pd.read_csv(os.path.join(folder_path, file2), sep="\t", encoding="utf-8")
for df in (df1, df2):
    df.columns = df.columns.str.strip()
    df['bidding_info_id'] = df['bidding_info_id'].astype(str).str.strip()
merged_df = pd.merge(df1, df2, on="bidding_info_id", how="left", suffixes=("_1", "_2"))
# ──────────────────────────────────────────────────────────────────────────────

merged_docs = [
    Document(
        page_content="\n".join(f"{col}: {row[col]}" for col in merged_df.columns),
        metadata={"bidding_info": row['bidding_info_id']}
    )
    for _, row in merged_df.iterrows()
]

# 토큰 및 배치 그룹화 설정
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
encoding = tiktoken.get_encoding("o200k_base")
def estimate_tokens(text): return len(encoding.encode(text))
def group_batches(docs, limit=150_000):
    batches, curr, cnt = [], [], 0
    for d in docs:
        t = estimate_tokens(d.page_content)
        if cnt + t > limit:
            batches.append(curr); curr, cnt = [d], t
        else:
            curr.append(d); cnt += t
    if curr: batches.append(curr)
    return batches

# FAISS 인덱스 로드/빌드
embedding = OpenAIEmbeddings()

# ─── 재빌드 판단 로직을 “메타+타임스탬프” 기반으로 수정했습니다 ──────────────────────
# (1) 원본 TSV 파일들 중 가장 최근 수정 시간을 구함
tsv_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.tsv')]
latest_data_ts = max(os.path.getmtime(p) for p in tsv_paths)

# (2) processed_files.json이 있으면, "code_version"과 "data_timestamp"를 읽어 비교
if os.path.exists(processed_files_path):
    with open(processed_files_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    saved_code_ver = meta.get("code_version", -1)
    saved_data_ts  = meta.get("data_timestamp", -1)
    rebuild = (saved_code_ver != CODE_VERSION) or (saved_data_ts != latest_data_ts)
else:
    # 메타 파일이 없으면 삭제하지 않고 기존 인덱스를 유지
    rebuild = False

# (3) 재빌드가 필요할 때만 index_dir 폴더를 삭제
if rebuild and os.path.exists(index_dir):
    shutil.rmtree(index_dir)
# ────────────────────────────────────────────────────────────────────────────────

if os.path.exists(index_dir) and os.path.exists(faiss_index_file):
    vectordb = FAISS.load_local(index_dir, embedding, allow_dangerous_deserialization=True)
    print("✅ 기존 인덱스 로드")
else:
    os.makedirs(index_dir, exist_ok=True)
    vectordb = None
    print("✅ 인덱스 초기화(캐시 없음)")

if vectordb is None:
    batches = group_batches(splitter.split_documents(merged_docs))
    vectordb = FAISS.from_documents(batches[0], embedding)
    for b in tqdm(batches[1:], desc="🔐 추가 임베딩"):
        vectordb.add_documents(b)
    vectordb.save_local(index_dir)
    # ─── 메타 기록 시 data_timestamp까지 함께 저장하도록 수정했습니다 ────────────────────
    with open(processed_files_path, 'w', encoding='utf-8') as f:
        json.dump({
            "code_version": CODE_VERSION,
            "data_timestamp": latest_data_ts
        }, f, ensure_ascii=False, indent=2)
    # ─────────────────────────────────────────────────────────────────────────────
    print("✅ 임베딩 완료 및 저장")

# PromptTemplate 정의
template = """
[시스템 지침]
1) source_documents>0 → <details> 토글 포함
2) source_documents=0 → 토글 블록 생략
3) 빈 토글 발생 시 제거
[끝]

{context}

질문: {question}

==출력 형식==
요구사항 고유번호: <값>
요구사항명: <값>
요구사항 분류: <값>
요구사항 정의: <값>
요구사항 세부내용: <값>
관련 요구사항: <값>
산출 정보: <값>
"""
formatted_prompt = PromptTemplate(input_variables=["context","question"], template=template)

# 체인 생성 함수
def create_chain(use_format=False):
    retriever = vectordb.as_retriever(search_kwargs={"k":3})
    return ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-4.1-mini", temperature=0),
        retriever=retriever,
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer"),
        return_source_documents=True,
        verbose=False,
        **({"combine_docs_chain_kwargs":{"prompt":formatted_prompt}} if use_format else {})
    )
default_chain = create_chain(False)
formatted_chain = create_chain(True)

# Gradio 챗봇 함수 (토글 로직 포함)
def chatbot_fn(message, state):
    state = state or []
    state.append({"role":"user","content":message})
    chain = formatted_chain if "출력" in message else default_chain
    result = chain.invoke({"question":message})
    answer = result.get("answer","")

    # 출처 토글 추가
    docs = result.get("source_documents", [])
    docs = [d for d in docs if d.metadata.get("bidding_info")]
    if docs:
        sources = "\n".join(f"- {d.metadata['bidding_info']}" for d in docs)
        answer += f"\n<details><summary>출처 내용 보기</summary>\n{sources}\n</details>"

    state.append({"role":"assistant","content":answer})
    return state, state, ""

# Gradio UI 블록
data_state = gr.State([])
with gr.Blocks() as demo:
    chat_display = gr.Chatbot(height=700, type="messages")
    txt = gr.Textbox(placeholder="질문을 입력하세요...")
    txt.submit(
        chatbot_fn,
        inputs=[txt, data_state],
        outputs=[chat_display, data_state, txt]
    )

demo.launch(inbrowser=True)
