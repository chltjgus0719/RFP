## 출처에 비딩넘버 뜨도록... 오후부터 수정중인 파일



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

# 환경 변수 로드
load_dotenv()

# 설정
folder_path = "sdf_sample"
index_dir = "faiss_index"
processed_files_path = os.path.join(index_dir, "processed_files.json")
faiss_index_file = os.path.join(index_dir, "index.faiss")
# 코드 버전 관리
CODE_VERSION = 2

# TSV 파일 자동 병합
tsv_files = [f for f in os.listdir(folder_path) if f.endswith('.tsv')]
if len(tsv_files) < 2:
    raise FileNotFoundError(f"폴더 '{folder_path}'에 최소 2개의 TSV 파일이 필요합니다: {tsv_files}")
file1, file2 = tsv_files[:2]
df1 = pd.read_csv(os.path.join(folder_path, file1), sep="\t", encoding="utf-8")
df2 = pd.read_csv(os.path.join(folder_path, file2), sep="\t", encoding="utf-8")
for df in (df1, df2):
    df.columns = df.columns.str.strip()
    df['bidding_info_id'] = df['bidding_info_id'].astype(str).str.strip()
merged_df = pd.merge(df1, df2, on="bidding_info_id", how="left", suffixes=("_1","_2"))
merged_docs = [
    Document(
        page_content='\n'.join(f"{col}: {row[col]}" for col in merged_df.columns),
        metadata={
            "bidding_info": row['bidding_info_id'],
            "source": f"{folder_path}/{file1}&{folder_path}/{file2}"
        }
    )
    for _, row in merged_df.iterrows()
]

# 텍스트 분할기 및 토큰 수 추정
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
encoding = tiktoken.get_encoding("o200k_base")

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

# 임베딩 객체 생성
embedding = OpenAIEmbeddings()

# 캐시 무효화 및 재빌드 조건
rebuild = True
if os.path.exists(processed_files_path):
    with open(processed_files_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    rebuild = meta.get('code_version') != CODE_VERSION
if rebuild and os.path.exists(index_dir):
    shutil.rmtree(index_dir)

# 인덱스 로드 또는 초기화
if os.path.exists(index_dir) and os.path.exists(faiss_index_file):
    vectordb = FAISS.load_local(index_dir, embedding, allow_dangerous_deserialization=True)
    print("✅ 기존 인덱스 로드")
else:
    os.makedirs(index_dir, exist_ok=True)
    vectordb = None
    print("✅ 인덱스 초기화(캐시 없음)")

# 임베딩 생성 또는 업데이트
if vectordb is None:
    batches = group_batches(splitter.split_documents(merged_docs))
    vectordb = FAISS.from_documents(batches[0], embedding)
    for b in tqdm(batches[1:], desc="🔐 추가 임베딩"):
        vectordb.add_documents(b)
    vectordb.save_local(index_dir)
    # 메타 기록
    with open(processed_files_path,'w',encoding='utf-8') as f:
        json.dump({"code_version": CODE_VERSION}, f, ensure_ascii=False, indent=2)
    print("✅ 임베딩 완료 및 저장")

# PromptTemplate 정의
template = """
[시스템 지침 시작]
1) source_documents가 1개 이상 조회되면 <details> 토글 포함
2) source_documents가 0개 조회되면 <details> 블록 절대 생략
3) 빈 <details> 블록이 생성되었거나 실제 텍스트가 없으면 해당 블록 전체 삭제
[시스템 지침 끝]

다음 병합된 문서를 참고하여 질문에 답해주세요.

{context}

질문: {question}

== 출력 형식 ==
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
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    return ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-4.1-mini", temperature=0),
        retriever=retriever,
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer"),
        return_source_documents=True,
        verbose=False,
        **({"combine_docs_chain_kwargs": {"prompt": formatted_prompt}} if use_format else {})
    )

# 체인 인스턴스
default_chain = create_chain(use_format=False)
formatted_chain = create_chain(use_format=True)

# Gradio 챗봇 함수
def chatbot_fn(message, history):
    history = history or []
    if "출력" in message:
        result = formatted_chain.invoke({"question": message})
        answer = result.get("answer", "")
        docs = [d for d in result.get("source_documents", []) if d.metadata.get("bidding_info")]
        if docs:
            sources = "\n".join(f"- {d.metadata['bidding_info']}" for d in docs)
            wrapped = f"<details><summary>출처 내용 보기</summary>\n{sources}\n</details>"
            answer = f"{answer}\n{wrapped}"
    else:
        result = default_chain.invoke({"question": message})
        answer = result.get("answer", "")
    history.append((message, answer))
    return history, history

with gr.Blocks() as demo:
    state = gr.State([])
    chat_display = gr.Chatbot()
    txt = gr.Textbox(placeholder="질문을 입력하세요...")
    txt.submit(chatbot_fn, [txt, state], [chat_display, state])

demo.launch(inbrowser=True)
