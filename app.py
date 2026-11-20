from datetime import datetime, timedelta, timezone
import json
import os
from google import genai
from google.genai import types
from PIL import Image
import requests
import streamlit as st

st.set_page_config(
    page_title="ありんこレシート家計簿＆振り返り",
    page_icon="🧾",
    layout="centered",
)

# 日本時間の今日の日付を取得
JST = timezone(timedelta(hours=9))
today_jst = datetime.now(JST).date()


# --- 秘密の鍵（Secrets）の読み込み ---
def get_secret(key, default=""):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)


GEMINI_API_KEY = get_secret("gemini_api_key")
GAS_WEBAPP_URL = get_secret("gas_webapp_url")

st.title("🧾 レシート家計簿 ＆ 1日の振り返り")

# サイドバーで手動入力もできるようにしておく（ローカルテスト用）
with st.sidebar:
    st.header("🔑 接続設定")
    if not GEMINI_API_KEY:
        GEMINI_API_KEY = st.text_input("Gemini API キー", type="password")
    else:
        st.success("✅ Gemini API 接続OK")

    if not GAS_WEBAPP_URL:
        GAS_WEBAPP_URL = st.text_input("GAS ウェブアプリ URL", type="password")
    else:
        st.success("✅ Googleシート連携 接続OK")

tab1, tab2 = st.tabs(["📸 レシート自動記録", "✍️ 1日の振り返り日記"])

# ==========================================================
# タブ1：レシート撮影・AI読み取り＆家計簿登録
# ==========================================================
with tab1:
    st.subheader("レシートを撮ってAIにおまかせ！")
    uploaded_file = st.file_uploader(
        "📷 カメラで撮影 または 画像を選択", type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="アップロードされたレシート", use_container_width=True)

        if st.button(
            "✨ AIでレシートを読み取る！",
            type="primary",
            use_container_width=True,
        ):
            if not GEMINI_API_KEY:
                st.error(
                    "👈 左メニュー（またはSecrets）に Gemini API キーを入れてね！"
                )
            else:
                with st.spinner("🤖 Geminiが品目と金額を読み取り中..."):
                    try:
                        client = genai.Client(api_key=GEMINI_API_KEY)
                        prompt = """
                        あなたは優秀な家計簿アシスタントです。
                        添付されたレシート画像から情報を読み取り、以下のJSON形式のみを出力してください。Markdownのコードブロック(```jsonなど)は含めないでください。

                        {
                          "date": "YYYY/MM/DD（読み取れない場合は空文字）",
                          "store": "店名（例：セブンイレブン、スーパーなど）",
                          "payment": "支払方法（現金、PayPay、クレジットカード、交通系ICなど。不明なら空文字）",
                          "items": [
                            {
                              "name": "品目名（割引や税込み調整があれば反映した実質額で）",
                              "category": "食費、日用品、交際費・飲み会、趣味・娯楽、交通費、美容・健康、その他のいずれか",
                              "price": 金額(整数),
                              "memo": "特記事項があれば（なければ空文字）"
                            }
                          ]
                        }
                        """
                        response = client.models.generate_content(
                            model="gemini-3.8-flash",
                            contents=[image, prompt],
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json"
                            ),
                        )
                        parsed = json.loads(response.text)
                        st.session_state["receipt_data"] = parsed
                        st.success("🎉 読み取り完了！内容を確認してね！")
                    except Exception as e:
                        st.error(f"読み取り中にエラーが起きたよ: {e}")

    # 解析結果の確認・編集・送信フォーム
    if "receipt_data" in st.session_state:
        rdata = st.session_state["receipt_data"]
        st.divider()
        st.markdown("### 📝 読み取り結果の確認・修正")

        col1, col2, col3 = st.columns(3)
        with col1:
            r_date = st.text_input(
                "購入日",
                value=rdata.get("date") or today_jst.strftime("%Y/%m/%d"),
            )
        with col2:
            r_store = st.text_input("店名", value=rdata.get("store", ""))
        with col3:
            r_pay = st.text_input("支払方法", value=rdata.get("payment", ""))

        st.caption(
            "👇 表の中身は直接タップして文字や金額を直せるよ！行の追加・削除もOK！"
        )
        edited_items = st.data_editor(
            rdata.get("items", []),
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "name": st.column_config.TextColumn("品目名", required=True),
                "category": st.column_config.SelectboxColumn(
                    "カテゴリ",
                    options=[
                        "食費",
                        "日用品",
                        "交際費・飲み会",
                        "趣味・娯楽",
                        "交通費",
                        "美容・健康",
                        "その他",
                    ],
                    required=True,
                ),
                "price": st.column_config.NumberColumn(
                    "金額 (円)", format="%d円", required=True
                ),
                "memo": st.column_config.TextColumn("メモ"),
            },
        )

        total_amount = sum(
            int(item.get("price", 0) or 0) for item in edited_items
        )
        st.metric("💰 合計金額", f"{total_amount:,} 円")

        if st.button(
            "📤 Googleスプレッドシートに記録する！",
            type="primary",
            use_container_width=True,
        ):
            if not GAS_WEBAPP_URL:
                st.error("👈 GAS ウェブアプリ URL が設定されていないよ！")
            else:
                with st.spinner("スプレッドシートへ送信中..."):
                    payload = {
                        "type": "receipt",
                        "date": r_date,
                        "store": r_store,
                        "payment": r_pay,
                        "items": edited_items,
                    }
                    try:
                        res = requests.post(
                            GAS_WEBAPP_URL, json=payload, timeout=15
                        )
                        if res.status_code == 200:
                            st.balloons()
                            st.success(
                                "✅ スプレッドシートの「家計簿明細」に保存したよ！！"
                            )
                            del st.session_state["receipt_data"]
                        else:
                            st.error(
                                f"送信エラー ({res.status_code}): {res.text}"
                            )
                    except Exception as e:
                        st.error(f"通信エラー: {e}")

# ==========================================================
# タブ2：1日の振り返り・日記記録
# ==========================================================
with tab2:
    st.subheader("✍️ 今日の振り返りログ")
    d_date = st.date_input("日付", value=today_jst)
    d_mood = st.select_slider(
        "今日の気分・コンディション",
        options=[
            "😫 ヘトヘト",
            "😐 ぼちぼち",
            "🙂 いい感じ",
            "😆 最高！",
            "🔥 絶好調！",
        ],
        value="🙂 いい感じ",
    )
    d_content = st.text_area(
        "今日あったこと・食べたもの・走った距離・思ったことなど",
        height=150,
        placeholder="例：仕事帰りに新宿から中野まで7kmウォーキングした！プロテインもバッチリ飲んだ！",
    )

    if st.button(
        "✨ AIの応援コメント付きで日記を保存！",
        type="primary",
        use_container_width=True,
    ):
        if not d_content.strip():
            st.warning("何か少しでも文章を書いてみてね！")
        elif not GAS_WEBAPP_URL:
            st.error("👈 GAS ウェブアプリ URL が設定されていないよ！")
        else:
            ai_reply = ""
            if GEMINI_API_KEY:
                with st.spinner("🤖 AIが今日の日記を読んでコメント作成中..."):
                    try:
                        client = genai.Client(api_key=GEMINI_API_KEY)
                        c_prompt = f"ユーザー（ありんこ）の今日の振り返り日記です。気分：{d_mood}\n内容：{d_content}\nこれに対して、親しみやすいフランクな口調で、労いやポジティブな一言コメントを2〜3文（100文字以内）で返してください。"
                        c_res = client.models.generate_content(
                            model="gemini-3.8-flash",
                            contents=c_prompt,
                        )
                        ai_reply = c_res.text.strip()
                    except Exception:
                        ai_reply = "今日もお疲れさま！"

            with st.spinner("スプレッドシートへ記録中..."):
                payload = {
                    "type": "diary",
                    "date": d_date.strftime("%Y/%m/%d"),
                    "mood": d_mood,
                    "content": d_content,
                    "ai_comment": ai_reply,
                }
                try:
                    res = requests.post(
                        GAS_WEBAPP_URL, json=payload, timeout=15
                    )
                    if res.status_code == 200:
                        st.success(
                            "✅ スプレッドシートの「1日の振り返り」に保存したよ！"
                        )
                        if ai_reply:
                            st.info(f"🤖 **AIからの一言：** {ai_reply}")
                    else:
                        st.error(f"送信エラー: {res.text}")
                except Exception as e:
                    st.error(f"通信エラー: {e}")