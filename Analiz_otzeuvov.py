import os
import json
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("API_BASE_URL", "https://api.openai.com/v1")
)
MODEL = os.getenv("MODEL_NAME", "gpt-4o-mini")

SYSTEM_PROMPT = """
Ты — строгий JSON-генератор для анализа отзывов о телефонах.
Возвращай ТОЛЬКО валидный JSON, пояснений и комментариев.
Структура ответа должна строго соответствовать схеме:
{
  "reviews": [
    {
      "id": <целое число>,
      "sentiment": "positive" | "negative" | "neutral",
      "topic": "<1-3 слова, основная тема отзыва>"
    }
  ]
}
"""


def call_llm(texts_batch):
    user_prompt = f"Проанализируй следующие отзывы о телефонах и верни результат в указанном строгом JSON-формате:\n{json.dumps(texts_batch, ensure_ascii=False, indent=2)}"

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )

        raw_content = response.choices[0].message.content.strip()

        if raw_content.startswith("```"):
            parts = raw_content.split("\n", 1)
            if len(parts) > 1:
                raw_content = parts[1]
            raw_content = raw_content.rsplit("```", 1)[0].strip()

        return json.loads(raw_content)

    except Exception as e:
        print(f"Error: {e}")
        return None


def rating_to_sentiment(rating):
    if rating >= 4:
        return "positive"
    elif rating <= 2:
        return "negative"
    else:
        return "neutral"


def main():
    input_file = "data.csv"

    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return

    df = pd.read_csv(input_file, encoding="utf-8")

    text_col = None
    for col in df.columns:
        if "text" in col.lower() or "review" in col.lower():
            text_col = col
            break

    if not text_col:
        print("Text column not found")
        return

    rating_col = None
    for col in df.columns:
        if "rating" in col.lower() or "star" in col.lower():
            rating_col = col
            break

    if "id" not in df.columns:
        df["id"] = range(1, len(df) + 1)

    df = df.head(500).reset_index(drop=True)

    results = []
    batch_size = 5

    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i + batch_size]
        batch = batch_df[["id", text_col]].rename(columns={text_col: "text"}).to_dict(orient="records")

        llm_response = call_llm(batch)

        if llm_response and "reviews" in llm_response:
            results.extend(llm_response["reviews"])

        time.sleep(2)

    os.makedirs("output", exist_ok=True)

    with open("output/result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    if rating_col and results:
        matches = 0
        for res in results:
            row = df[df["id"] == res["id"]]
            if not row.empty and "sentiment" in res:
                true_rating = row[rating_col].values[0]
                true_sent = rating_to_sentiment(true_rating)
                if res["sentiment"] == true_sent:
                    matches += 1
        accuracy = matches / len(results) * 100
        print(f"Accuracy: {accuracy:.1f}%")


if __name__ == "__main__":
    main()
