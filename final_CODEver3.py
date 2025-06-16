import os
import json
import shutil
import pandas as pd
from dotenv import load_dotenv

# 원래 쓰시던 패키지 경로로 복원
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

# ─── 환경 변수 및 경로 설정 ─────────────────────────────────────────────────
load_dotenv()
folder_path = "C:\\Users\\gram\\Desktop\\langchain_trytry\\sdf_sample"

index_dir = "C:\\Users\\gram\\Desktop\\langchain_trytry\\faiss_index"
processed_files_path = os.path.join(index_dir, "processed_files.json")
faiss_index_file = os.path.join(index_dir, "index.faiss")
# ───────────────────────────────────────────────────────────────────────────────

CODE_VERSION = 3

# TSV 파일 자동 병합
tsv_files = [f for f in os.listdir(folder_path) if f.endswith('.tsv')]
if len(tsv_files) < 2:
    raise FileNotFoundError(f"폴더 '{folder_path}'에 최소 2개의 TSV 파일이 필요합니다: {tsv_files}")
file1, file2 = tsv_files[:2]

# ─── 병합 로직 ────────────────────────────────────────────────────────────────
df1 = pd.read_csv(os.path.join(folder_path, file1), sep="\t", encoding="utf-8")
df2 = pd.read_csv(os.path.join(folder_path, file2), sep="\t", encoding="utf-8")
for df in (df1, df2):
    df.columns = df.columns.str.strip()
    df['bidding_info_id'] = df['bidding_info_id'].astype(str).str.strip()

merged_df = pd.merge(df1, df2, on="bidding_info_id", how="left", suffixes=("_1", "_2"))
# ───────────────────────────────────────────────────────────────────────────────

# 각 행을 Document 객체로 변환
merged_docs = [
    Document(
        page_content="\n".join(f"{col}: {row[col]}" for col in merged_df.columns),
        metadata={"bidding_info": row['bidding_info_id']}
    )
    for _, row in merged_df.iterrows()
]

# 토큰 수 계산 및 배치 그룹화

# ─── 텍스트 분할기 설정 (수정된 부분) ─────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=[
        "\n"  # 연속된 빈 줄(단락 구분)으로만 분할
    ]
)
# ───────────────────────────────────────────────────────────────────────────────
encoding = tiktoken.get_encoding("o200k_base")  # 원래대로 "o200k_base" 사용
def estimate_tokens(text):
    return len(encoding.encode(text))

def group_batches(docs, limit=150_000):
    batches, curr, cnt = [], [], 0
    for d in docs:
        t = estimate_tokens(d.page_content)
        if cnt + t > limit:
            batches.append(curr)
            curr, cnt = [d], t
        else:
            curr.append(d)
            cnt += t
    if curr:
        batches.append(curr)
    return batches

# FAISS 인덱스 로드/빌드
embedding = OpenAIEmbeddings()

# ─── 재빌드 판단 로직 ─────────────────────────────────────────────────────────
tsv_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.tsv')]
latest_data_ts = max(os.path.getmtime(p) for p in tsv_paths)

if os.path.exists(processed_files_path):
    with open(processed_files_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    saved_code_ver = meta.get("code_version", -1)
    saved_data_ts  = meta.get("data_timestamp", -1)
    rebuild = (saved_code_ver != CODE_VERSION) or (saved_data_ts != latest_data_ts)
else:
    rebuild = False

if rebuild and os.path.exists(index_dir):
    shutil.rmtree(index_dir)
# ───────────────────────────────────────────────────────────────────────────────

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
    with open(processed_files_path, 'w', encoding='utf-8') as f:
        json.dump({
            "code_version": CODE_VERSION,
            "data_timestamp": latest_data_ts
        }, f, ensure_ascii=False, indent=2)
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


# ─── Gradio UI 설정 ─────────────────────────────────────────────────────────────
with gr.Blocks() as demo:
    # State 컴포넌트를 반드시 Blocks 내부에서 선언해야 합니다
    data_state = gr.State([])

    chat_display = gr.Chatbot(height=700, type="messages")
    txt = gr.Textbox(placeholder="질문을 입력하세요...")

    # 챗봇 함수 정의
    def chatbot_fn(message, state):
        state = state or []
        state.append({"role":"user","content":message})
        chain = formatted_chain if "출력" in message else default_chain

        # 원래 쓰시던 invoke() 방식으로 호출
        result = chain.invoke({"question": message})
        answer = result.get("answer", "")

        # 출처 토글 추가
        docs = result.get("source_documents", [])
        docs = [d for d in docs if d.metadata.get("bidding_info")]
        if docs:
            sources = "\n".join(f"- {d.metadata['bidding_info']}" for d in docs)
            answer += f"\n<details><summary>출처 내용 보기</summary>\n{sources}\n</details>"

        state.append({"role":"assistant","content":answer})
        return state, state, ""  # (chat_display, data_state, txt 순서로 리턴)

    # Gradio 컴포넌트를 연결
    txt.submit(
        fn=chatbot_fn,
        inputs=[txt, data_state],
        outputs=[chat_display, data_state, txt]
    )

    # 실행
    demo.launch(inbrowser=True)
