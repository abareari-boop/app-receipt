from datetime import datetime, timedelta, timezone
import io
import json
import os
import time
from google import genai
from google.genai import types
import pandas as pd
from PIL import Image, ImageOps
import requests
import streamlit as st

st.set_page_config(
    page_title="ありんこレシート家計簿＆振り返り",
    page_icon="🧾",
    layout="centered",
)

JST = timezone(timedelta(hours=9))
today_jst = datetime.now(JST).date()


def get_secret(key, default=""):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)


GEMINI_API_KEY = get_secret("gemini_api_key")
GAS_WEBAPP_URL = get_secret("gas_webapp_url")
APP_PASSWORD = get_secret("app_password")

# ==========================================================
# 🔒 合言葉（パスワード）ロック機能
# ==========================================================
if APP_PASSWORD:
    if not st.session_state.get("authenticated", False):
        st.title("🔒 ありんこ専用ロック")
        pw_input = st.text_input("合言葉（パスワード）", type="password")
        if st.button("🔓 ロック解除", type="primary", use_container_width=True):
            if pw_input == APP_PASSWORD:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("合言葉が違うよ！")
        st.stop()

st.title("🧾 レシート家計簿 ＆ 振り返り")

with st.sidebar:
    st.header("⚙️ 設定・目標予算")
    monthly_budget = st.number_input(
        "💰 今月の目標予算（円）",
        min_value=10000,
        max_value=1000000,
        value=100000,
        step=5000,
    )
    model_choice = st.selectbox(
        "🤖 AIモデル設定",
        [
            "🚀 自動リレー（混雑回避・おすすめ）",
            "gemini-3.8-flash-lite",
            "gemini-3.8-flash",
            "gemini-2.5-flash-lite",
        ],
        index=0,
    )
    st.divider()
    if GEMINI_API_KEY:
        st.success("✅ Gemini API 接続OK")
    else:
        GEMINI_API_KEY = st.text_input("Gemini API キー", type="password")

    if GAS_WEBAPP_URL:
        st.success("✅ Googleシート連携 接続OK")
    else:
        GAS_WEBAPP_URL = st.text_input("GAS ウェブアプリ URL", type="password")

    if APP_PASSWORD:
        if st.button("🔒 画面をロックする", use_container_width=True):
            st.session_state["authenticated"] = False
            st.rerun()


# ==========================================================
# ⚡ 爆速化＆503混雑回避ヘルパー
# ==========================================================
def compress_receipt_image(uploaded_file, max_width=1024):
    img = Image.open(uploaded_file)
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if w > max_width:
        new_h = int(h * (max_width / w))
        img = img.resize((max_width, new_h), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80, optimize=True)
    buf.seek(0)
    return Image.open(buf)


def call_gemini_with_fallback(contents, is_json=False):
    client = genai.Client(api_key=GEMINI_API_KEY)

    # ありんこのAPIキーで実際に使えるモデル一覧をサーバーから直接取得！
    available_models = []
    try:
        for m in client.models.list():
            name = m.name.replace("models/", "")
            # 画像やテキスト生成（generateContent）に対応しているflash系を優先収集
            actions = getattr(m, "supported_actions", []) or []
            if "generateContent" in actions and "flash" in name.lower():
                available_models.append(name)
    except Exception:
        pass

    # 一覧取得できたモデルを優先し、念のため既知のモデルも後ろに並べる
    candidates = []
    if "自動リレー" not in model_choice:
        candidates.append(model_choice)
    for m in available_models + ["gemini-3.8-flash", "gemini-3-flash-preview", "gemini-2.5-flash"]:
        if m not in candidates:
            candidates.append(m)

    cfg_kwargs = {}
    if is_json:
        cfg_kwargs["response_mime_type"] = "application/json"

    last_err = None
    for idx, model_name in enumerate(candidates):
        try:
            cfg = types.GenerateContentConfig(**cfg_kwargs)
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=cfg,
            )
            return response.text, model_name
        except Exception as e:
            last_err = e
            err_str = str(e)
            # 404（非対応）・503（混雑）・429（制限）なら自動で次の利用可能モデルへスキップ！
            if any(code in err_str for code in ["404", "NOT_FOUND", "503", "UNAVAILABLE", "429"]):
                time.sleep(1.0)
                continue
            raise e
    raise last_err


def analyze_receipt_with_gemini(image):
    prompt = """
    レシート画像から情報を抽出し、以下のJSON形式のみを出力してください。Markdown記号は不要です。
    値引きやクーポンがある場合は該当品目から引くか「値引き」項目をマイナス金額で入れ、合計が支払総額と一致するよう調整してください。
    {"date":"YYYY/MM/DD","store":"店名","payment":"支払方法","items":[{"name":"品目名","category":"食費|日用品|交際費・飲み会|趣味・娯楽|交通費|美容・健康|その他","price":0,"memo":""}]}
    """
    raw_text, used_model = call_gemini_with_fallback([image, prompt], is_json=True)
    return json.loads(raw_text), used_model


def post_to_gas(payload):
    if not GAS_WEBAPP_URL:
        st.error("GAS ウェブアプリ URL が設定されていないよ！")
        return False
    try:
        res = requests.post(GAS_WEBAPP_URL, json=payload, timeout=15)
        if res.status_code == 200:
            return True
        st.error(f"送信エラー ({res.status_code}): {res.text}")
    except Exception as e:
        st.error(f"通信エラー: {e}")
    return False


@st.cache_data(ttl=60, show_spinner=False)
def fetch_from_gas(url):
    if not url:
        return None
    try:
        res = requests.get(url, timeout=15)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return None


tab1, tab2, tab3, tab4 = st.tabs(
    ["📸 レシート撮影", "⚡ ワンタップ記録", "✍️ 1日の振り返り", "📊 今月の集計"]
)

# ==========================================================
# タブ1：レシート撮影・爆速AI読み取り＆家計簿登録
# ==========================================================
with tab1:
    st.subheader("レシートを撮って爆速記録！")
    uploaded_file = st.file_uploader(
        "📷 カメラで撮影 または 画像を選択", type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        light_image = compress_receipt_image(uploaded_file)
        st.image(light_image, width=280)

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            auto_save_btn = st.button(
                "⚡ 読み取って即シートへ保存！",
                type="primary",
                use_container_width=True,
                help="確認画面を飛ばして、AI読み取りからスプレッドシート記帳まで一発で完了します！",
            )
        with col_btn2:
            preview_btn = st.button(
                "🔍 まず内容を確認・編集する",
                use_container_width=True,
            )

        if auto_save_btn or preview_btn:
            if not GEMINI_API_KEY:
                st.error("Gemini API キーを設定してね！")
            else:
                with st.spinner("⚡ 超高速スキャン中..."):
                    try:
                        parsed, used_model = analyze_receipt_with_gemini(light_image)
                        if auto_save_btn:
                            payload = {
                                "type": "receipt",
                                "date": parsed.get("date") or today_jst.strftime("%Y/%m/%d"),
                                "store": parsed.get("store", ""),
                                "payment": parsed.get("payment", ""),
                                "items": parsed.get("items", []),
                            }
                            if post_to_gas(payload):
                                total_p = sum(int(i.get("price", 0) or 0) for i in parsed.get("items", []))
                                st.balloons()
                                st.success(
                                    f"🎉 一発記録完了！【{parsed.get('store', '店舗')}】合計 {total_p:,}円（{len(parsed.get('items', []))}品目）を保存したよ！（使用AI: {used_model}）"
                                )
                                fetch_from_gas.clear()
                        else:
                            st.session_state["receipt_data"] = parsed
                            st.success(f"🎉 読み取り完了！（使用AI: {used_model}）下で修正できるよ！")
                    except Exception as e:
                        st.error(f"読み取りエラー: {e}")

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
            with st.spinner("スプレッドシートへ送信中..."):
                payload = {
                    "type": "receipt",
                    "date": r_date,
                    "store": r_store,
                    "payment": r_pay,
                    "items": edited_items,
                }
                if post_to_gas(payload):
                    st.balloons()
                    st.success("✅ スプレッドシートに保存したよ！！")
                    del st.session_state["receipt_data"]
                    fetch_from_gas.clear()

# ==========================================================
# タブ2：レシートなし・ワンタップクイック記録
# ==========================================================
with tab2:
    st.subheader("⚡ レシートなし・ワンタップ記録")
    presets = [
        ("🥤 自販機・コンビニ飲料", "コンビニ/自販機", "食費", 160, "交通系IC"),
        ("🚃 交通系ICチャージ", "駅券売機", "交通費", 3000, "現金"),
        ("🍜 ランチ・ラーメン外食", "飲食店", "食費", 1000, "PayPay"),
        ("🍻 飲み会・居酒屋", "居酒屋", "交際費・飲み会", 5000, "現金"),
        ("💪 プロテイン・サプリ補充", "ネット通販", "美容・健康", 4000, "クレジットカード"),
        ("🎨 同人誌・クリエイター支援", "BOOTH/FANBOX", "趣味・娯楽", 1000, "クレジットカード"),
    ]

    p_cols = st.columns(2)
    for idx, (label, store, cat, price, pay) in enumerate(presets):
        with p_cols[idx % 2]:
            if st.button(
                f"{label}\n({price:,}円)",
                key=f"preset_{idx}",
                use_container_width=True,
            ):
                payload = {
                    "type": "receipt",
                    "date": today_jst.strftime("%Y/%m/%d"),
                    "store": store,
                    "payment": pay,
                    "items": [
                        {
                            "name": label.split(" ", 1)[1],
                            "category": cat,
                            "price": price,
                            "memo": "ワンタップ登録",
                        }
                    ],
                }
                with st.spinner("記録中..."):
                    if post_to_gas(payload):
                        st.toast(f"✅ 「{label}」({price:,}円) を記録したよ！")
                        fetch_from_gas.clear()

    st.divider()
    st.markdown("### ✍️ 自由入力（手打ちサクッと登録）")
    with st.form("quick_manual_form", clear_on_submit=True):
        q_date = st.date_input("日付", value=today_jst)
        qc1, qc2 = st.columns(2)
        with qc1:
            q_store = st.text_input("店名・場所", placeholder="例：松屋、Amazon")
            q_name = st.text_input("品目名", placeholder="例：牛めし並、単3電池")
        with qc2:
            q_cat = st.selectbox(
                "カテゴリ",
                [
                    "食費",
                    "日用品",
                    "交際費・飲み会",
                    "趣味・娯楽",
                    "交通費",
                    "美容・健康",
                    "その他",
                ],
            )
            q_price = st.number_input("金額（円）", min_value=1, value=500, step=10)
        q_pay = st.selectbox(
            "支払方法", ["PayPay", "交通系IC", "クレジットカード", "現金", "その他"]
        )
        if st.form_submit_button(
            "📤 この内容で記録する", type="primary", use_container_width=True
        ):
            if not q_name.strip():
                st.warning("品目名を入れてね！")
            else:
                payload = {
                    "type": "receipt",
                    "date": q_date.strftime("%Y/%m/%d"),
                    "store": q_store or "手入力",
                    "payment": q_pay,
                    "items": [
                        {
                            "name": q_name,
                            "category": q_cat,
                            "price": int(q_price),
                            "memo": "",
                        }
                    ],
                }
                if post_to_gas(payload):
                    st.success(f"✅ 「{q_name}」({q_price:,}円) を保存したよ！")
                    fetch_from_gas.clear()

# ==========================================================
# タブ3：1日の振り返り・ランニング＆体重記録
# ==========================================================
with tab3:
    st.subheader("✍️ 今日の振り返り ＆ コンディション記録")
    d_date = st.date_input("日付", value=today_jst, key="diary_date")
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

    dc1, dc2 = st.columns(2)
    with dc1:
        d_dist = st.number_input(
            "🏃‍♂️ ラン・ウォーキング距離 (km)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.5,
            format="%.1f",
        )
    with dc2:
        d_weight = st.number_input(
            "⚖️ 今日の体重 (kg) ※未計測は0",
            min_value=0.0,
            max_value=150.0,
            value=0.0,
            step=0.1,
            format="%.1f",
        )

    d_content = st.text_area(
        "今日あったこと・食べたもの・思ったこと",
        height=130,
        placeholder="例：仕事帰りに新宿から中野まで7km歩いた！プロテインもバッチリ！",
    )

    if st.button(
        "✨ AIの応援コメント付きで保存！",
        type="primary",
        use_container_width=True,
    ):
        if not d_content.strip() and d_dist == 0 and d_weight == 0:
            st.warning("日記か距離・体重のどれかを入力してね！")
        else:
            ai_reply = ""
            if GEMINI_API_KEY:
                with st.spinner("🤖 AIコメント作成中..."):
                    try:
                        extra_info = []
                        if d_dist > 0:
                            extra_info.append(f"運動距離: {d_dist}km")
                        if d_weight > 0:
                            extra_info.append(f"体重: {d_weight}kg")
                        c_prompt = (
                            f"ユーザー（ありんこ・東京マラソン完走やダイエットも頑張っている）の今日の記録です。\n"
                            f"気分：{d_mood} / {' / '.join(extra_info)}\n"
                            f"内容：{d_content}\n"
                            f"親しみやすいフランクな口調で、労いやポジティブな一言コメントを2文（100文字以内）で返してください。"
                        )
                        ai_reply, _ = call_gemini_with_fallback(c_prompt, is_json=False)
                        ai_reply = ai_reply.strip()
                    except Exception:
                        ai_reply = "今日もお疲れさま！ゆっくり休んでね！"

            with st.spinner("スプレッドシートへ記録中..."):
                payload = {
                    "type": "diary",
                    "date": d_date.strftime("%Y/%m/%d"),
                    "mood": d_mood,
                    "content": d_content,
                    "ai_comment": ai_reply,
                    "distance": d_dist if d_dist > 0 else None,
                    "weight": d_weight if d_weight > 0 else None,
                }
                if post_to_gas(payload):
                    st.success("✅ 「1日の振り返り」に保存したよ！")
                    if ai_reply:
                        st.info(f"🤖 **AIからの一言：** {ai_reply}")
                    fetch_from_gas.clear()

# ==========================================================
# タブ4：今月の集計＆グラフ・過去ログ閲覧
# ==========================================================
with tab4:
    st.subheader("📊 家計簿集計 ＆ 振り返りログ")
    if st.button("🔄 最新データに更新", use_container_width=True):
        fetch_from_gas.clear()
        st.rerun()

    gas_data = fetch_from_gas(GAS_WEBAPP_URL)
    if not gas_data or gas_data.get("status") != "success":
        st.info(
            "💡 スプレッドシートのデータを読み込めなかったよ！GASに `doGet` を追加して「新バージョン」でデプロイしたか確認してね！"
        )
    else:
        receipts = gas_data.get("receipts", [])
        diaries = gas_data.get("diaries", [])

        if receipts:
            df = pd.DataFrame(receipts)
            df["ym"] = df["date"].str.slice(0, 7)
            ym_list = sorted(df["ym"].dropna().unique().tolist(), reverse=True)
            current_ym = today_jst.strftime("%Y/%m")
            default_idx = ym_list.index(current_ym) if current_ym in ym_list else 0

            selected_ym = st.selectbox("📅 表示する年月", ym_list, index=default_idx)
            month_df = df[df["ym"] == selected_ym]

            total_spent = int(month_df["price"].sum())
            remain = monthly_budget - total_spent

            m1, m2 = st.columns(2)
            with m1:
                st.metric(f"💸 {selected_ym} の支出合計", f"{total_spent:,} 円")
            with m2:
                st.metric(
                    "💰 今月の残り予算",
                    f"{remain:,} 円",
                    delta=f"予算 {monthly_budget:,}円",
                )

            ratio = min(1.0, max(0.0, total_spent / max(1, monthly_budget)))
            st.progress(ratio, text=f"予算消化率：{int(ratio * 100)}%")

            st.markdown("### 📈 カテゴリ別支出")
            cat_summary = (
                month_df.groupby("category")["price"]
                .sum()
                .sort_values(ascending=False)
            )
            st.bar_chart(cat_summary)

            for cat_name, val in cat_summary.items():
                st.write(f"・**{cat_name}**：{int(val):,} 円")

            st.divider()
            st.markdown("### 🧾 最近の明細一覧（最新20件）")
            show_df = month_df[
                ["date", "store", "name", "category", "price", "payment"]
            ].iloc[::-1].head(20)
            show_df.columns = ["日付", "店名", "品目", "カテゴリ", "金額", "支払"]
            st.dataframe(show_df, use_container_width=True, hide_index=True)
        else:
            st.info("まだ家計簿データがないよ！レシートを登録してみよう！")

        if diaries:
            st.divider()
            st.markdown("### 📖 最近の振り返りログ（最新7日分）")
            for d in reversed(diaries[-7:]):
                badges = [d.get("mood", "")]
                if d.get("distance"):
                    badges.append(f"🏃‍♂️ {d['distance']}km")
                if d.get("weight"):
                    badges.append(f"⚖️ {d['weight']}kg")
                with st.container(border=True):
                    st.markdown(f"**📅 {d['date']}** ｜ {' / '.join(badges)}")
                    if d.get("content"):
                        st.write(d["content"])
                    if d.get("ai_comment"):
                        st.caption(f"🤖 {d['ai_comment']}")